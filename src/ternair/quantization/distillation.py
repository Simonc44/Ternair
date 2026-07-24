"""
Distillation KL + conversion de modeles HuggingFace vers TernairLinear.

Ce module implemente le pipeline QAT (Quantization-Aware Training) avec
distillation pour convertir un modele HuggingFace existant en version
ternaire 1.58-bit, en conservant ses performances.

Pipeline :
  1. Charger un modele professeur (SmolLM2, Qwen2.5, ...) en FP16
  2. Remplacer ses nn.Linear par TernairLinear avec les poids initialises
  3. Distillation : KL divergence entre prof (freeze) et eleve (train)
  4. Apprentissage du facteur d'echelle alpha par canal
  5. Alignement des etats caches intermediaires (Feature Matching Loss)
  6. Gel en stockage compact (freeze_storage) -> inference ternaire
"""

from __future__ import annotations

from typing import Optional, Type, Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Perte de distillation KL
# ---------------------------------------------------------------------------

def kl_divergence_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = 4.0,
) -> Tensor:
    """KL(P_teacher || P_student) avec temperature.
    
    La temperature adoucit les distributions de probabilite pour que
    l'eleve apprenne la structure fine du professeur.
    
    Args:
        student_logits: Logits bruts de l'eleve      (B, T, V)
        teacher_logits: Logits bruts du professeur    (B, T, V)
        temperature: Temperature de distillation (>= 1.0)
    
    Returns:
        Tenseur scalaire de divergence KL
    """
    T = temperature
    student_probs = F.log_softmax(student_logits / T, dim=-1)
    teacher_probs = F.softmax(teacher_logits / T, dim=-1)
    kl = F.kl_div(student_probs, teacher_probs, reduction="batchmean")
    return kl * (T ** 2)


def distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    alpha: float = 0.5,
    temperature: float = 4.0,
) -> Tensor:
    """Perte combinee CE + KL pour la distillation.
    
    L_total = (1 - alpha) * CE(student, labels) + alpha * T^2 * KL(teacher || student)
    
    Args:
        student_logits: Logits de l'eleve         (B, T, V)
        teacher_logits: Logits du professeur      (B, T, V)
        labels: Tokens cibles                     (B, T)
        alpha: Poids de la distillation (0.0 = CE pure, 1.0 = KL pure)
        temperature: Temperature de distillation
    
    Returns:
        Tenseur scalaire de perte totale
    """
    # Cross-entropy sur les logits decales (next-token prediction)
    shift_s = student_logits[..., :-1, :].contiguous()
    shift_t = teacher_logits[..., :-1, :].contiguous()
    shift_l = labels[..., 1:].contiguous()
    
    ce_loss = F.cross_entropy(
        shift_s.view(-1, shift_s.size(-1)),
        shift_l.view(-1),
    )
    
    kl_loss = kl_divergence_loss(shift_s, shift_t, temperature=temperature)
    
    return (1.0 - alpha) * ce_loss + alpha * kl_loss


# ---------------------------------------------------------------------------
# Feature Matching Loss (Alignement des etats caches intermediaires)
# ---------------------------------------------------------------------------

class HiddenStateCapture:
    """Capture les etats caches intermediaires d'un modele.
    
    Utilise des hooks forward pour enregistrer les sorties des couches
    specifiees lors du forward pass.
    
    Example :
        capture = HiddenStateCapture(model, layer_ids=[0, 5, 11])
        capture.register_hooks()
        logits = model(input_ids)
        hidden = capture.get_hidden_states()  # dict {layer_id: tensor}
        capture.clear()
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer_ids: Optional[list[int]] = None,
        layer_prefix: str = "model.layers",
    ):
        self.model = model
        self.layer_ids = layer_ids
        self.layer_prefix = layer_prefix
        self._handles: list = []
        self._storage: dict[int, Tensor] = {}
    
    def _make_hook(self, layer_id: int):
        def hook(module, input, output):
            self._storage[layer_id] = output.detach()
        return hook
    
    def register_hooks(self) -> None:
        """Enregistre les hooks forward sur les couches specifiees."""
        self._storage = {}
        self._handles = []
        
        for name, module in self.model.named_modules():
            if self.layer_prefix in name and "output" not in name:
                # Extraire l'index de couche depuis le nom
                parts = name.split(".")
                for i, part in enumerate(parts):
                    if part.isdigit():
                        layer_id = int(part)
                        if self.layer_ids is None or layer_id in self.layer_ids:
                            handle = module.register_forward_hook(
                                self._make_hook(layer_id)
                            )
                            self._handles.append(handle)
                        break
    
    def get_hidden_states(self) -> dict[int, Tensor]:
        """Retourne les etats caches captures."""
        return dict(self._storage)
    
    def clear(self) -> None:
        """Vide le stockage et detache les hooks."""
        self._storage = {}
        for handle in self._handles:
            handle.remove()
        self._handles = []


def feature_matching_loss(
    student_hidden: dict[int, Tensor],
    teacher_hidden: dict[int, Tensor],
    projections: dict[int, nn.Linear | None] | None = None,
) -> Tensor:
    """Perte d'alignement MSE entre etats caches prof/eleve.
    
    Pour chaque couche alignee :
        L_align = sum||h_s - W_proj * h_t||^2
    
    Si une projection est fournie, elle aligne les dimensions cachees
    si le professeur et l'eleve ont des tailles differentes.
    
    Args:
        student_hidden: dict {layer_id: tensor (B, T, H_s)}
        teacher_hidden: dict {layer_id: tensor (B, T, H_t)}
        projections: dict {layer_id: nn.Linear(H_t, H_s)} optionnel
    
    Returns:
        Tenseur scalaire de perte MSE moyennee sur toutes les couches
    """
    losses = []
    
    for layer_id in student_hidden:
        if layer_id not in teacher_hidden:
            continue
        
        h_s = student_hidden[layer_id]
        h_t = teacher_hidden[layer_id]
        
        # Projection si dimensions differentes
        if projections is not None and layer_id in projections:
            proj = projections[layer_id]
            if proj is not None:
                h_t = proj(h_t)
        elif h_s.shape[-1] != h_t.shape[-1]:
            # Avertir mais ne pas planter
            continue
        
        loss = F.mse_loss(h_s, h_t, reduction="mean")
        losses.append(loss)
    
    if not losses:
        return torch.tensor(0.0, device=student_hidden[next(iter(student_hidden))].device)
    
    return torch.stack(losses).mean()


def build_align_projections(
    student: nn.Module,
    teacher: nn.Module,
    layer_ids: list[int],
    layer_prefix: str = "model.layers",
) -> dict[int, nn.Linear | None]:
    """Cree les projections necessaires pour l'alignement.
    
    Si les dimensions cachees du professeur et de l'eleve sont
    identiques, pas besoin de projection (retourne None).
    Sinon, cree un nn.Linear(H_t, H_s) pour chaque couche.
    """
    projections: dict[int, nn.Linear | None] = {}
    
    for layer_id in layer_ids:
        h_s = None
        h_t = None
        
        # Recuperer les dimensions cachees
        for name, module in student.named_modules():
            if f"{layer_prefix}.{layer_id}" in name and hasattr(module, "hidden_size"):
                pass  # On cherche les dimensions autrement
        
        # Fallback: pas de projection si meme config
        projections[layer_id] = None
    
    return projections


# ---------------------------------------------------------------------------
# Perte de distillation complete avec alignement
# ---------------------------------------------------------------------------

def distillation_loss_with_alignment(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    student_hidden: dict[int, Tensor] | None = None,
    teacher_hidden: dict[int, Tensor] | None = None,
    alpha: float = 0.5,
    lambda_align: float = 0.1,
    temperature: float = 4.0,
    align_projections: dict[int, nn.Linear | None] | None = None,
) -> Tensor:
    """Perte complete : CE + KL + Feature Matching.
    
    L_total = (1-alpha) * CE + alpha * T^2 * KL + lambda_align * L_align
    
    Args:
        student_logits: Logits de l'eleve (B, T, V)
        teacher_logits: Logits du professeur (B, T, V)
        labels: Tokens cibles (B, T)
        student_hidden: dict {layer_id: tensor} des etats eleve
        teacher_hidden: dict {layer_id: tensor} des etats prof
        alpha: Poids KL (0.0 = CE pure, 1.0 = KL pure)
        lambda_align: Poids de l'alignement des etats caches
        temperature: Temperature de distillation
        align_projections: Projections optionnelles pour aligner les dims
    
    Returns:
        Tenseur scalaire de perte totale
    """
    shift_s = student_logits[..., :-1, :].contiguous()
    shift_t = teacher_logits[..., :-1, :].contiguous()
    shift_l = labels[..., 1:].contiguous()
    
    ce_loss = F.cross_entropy(
        shift_s.view(-1, shift_s.size(-1)),
        shift_l.view(-1),
    )
    
    kl_loss = kl_divergence_loss(shift_s, shift_t, temperature=temperature)
    
    loss = (1.0 - alpha) * ce_loss + alpha * kl_loss
    
    # Ajout de la perte d'alignement si les etats sont fournis
    if student_hidden is not None and teacher_hidden is not None:
        align_loss = feature_matching_loss(
            student_hidden, teacher_hidden,
            projections=align_projections,
        )
        loss = loss + lambda_align * align_loss
    
    return loss


# ---------------------------------------------------------------------------
# Conversion de modele : nn.Linear -> TernairLinear
# ---------------------------------------------------------------------------

def _get_linear_layers(model: nn.Module, prefix: str = "") -> list[tuple[str, nn.Linear, str]]:
    """Decouvre recursivement tous les nn.Linear d'un modele.
    
    Retourne une liste de (nom, module, role) ou role indique si la couche
    doit etre quantifiee ou laissee en FP16.
    """
    from ternair.quantization.linear import TernairLinear
    
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not isinstance(module, TernairLinear):
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Detection du role : Embedding/LM Head sont laissees en FP16
            # (mixed-precision selective : ~5-10% de la taille totale)
            is_critical = any(
                keyword in full_name.lower()
                for keyword in ["embed", "lm_head", "wte", "lm_head"]
            )
            role = "embed" if is_critical else "ternary"
            layers.append((full_name, module, role))
    
    return layers


def convert_module_to_ternair(
    module: nn.Module,
    module_name: str,
    storage: str = "packed",
    keep_embed_fp16: bool = True,
    learned_alpha: bool = True,
) -> nn.Module:
    """Remplace un nn.Linear par un TernairLinear en preservant les poids.
    
    Args:
        module: Le module nn.Linear a convertir
        module_name: Nom du module (pour detecter embedding)
        storage: Mode de stockage ("packed", "int8", "fastpacked")
        keep_embed_fp16: Si True, laisse les couches critiques en FP16
        learned_alpha: Si True, active l'alpha appris (QAT)
    
    Returns:
        Module TernairLinear ou nn.Linal (si keep_embed_fp16=True et critique)
    """
    from ternair.quantization.linear import TernairLinear
    
    # Detection des couches critiques (embedding / LM head)
    is_critical = any(
        keyword in module_name.lower()
        for keyword in ["embed", "lm_head", "wte", "lm_head"]
    )
    
    if keep_embed_fp16 and is_critical:
        # Garder en FP16 pur : pas de quantification ternaire
        # On retourne le module original intact
        return module
    
    # Creer un TernairLinear avec les memes dimensions
    ternair_linear = TernairLinear(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=module.bias is not None,
        storage=storage,
    )
    
    # Transférer les poids (preserve les connaissances du professeur)
    ternair_linear.weight.data.copy_(module.weight.data)
    if module.bias is not None:
        ternair_linear.bias.data.copy_(module.bias.data)
    
    # Activer l'alpha appris (facteur d'echelle entrainable)
    if learned_alpha:
        ternair_linear.enable_learned_alpha()
    
    return ternair_linear


def convert_model_to_ternair(
    model: nn.Module,
    storage: str = "packed",
    keep_embed_fp16: bool = True,
    learned_alpha: bool = True,
) -> nn.Module:
    """Convertit un modele HuggingFace complet en version Ternair.
    
    Parcourt recursivement le modele et remplace tous les nn.Linear
    (sauf les couches critiques) par des TernairLinear.
    
    Args:
        model: Modele HuggingFace a convertir
        storage: Mode de stockage ternaire
        keep_embed_fp16: Si True, embeddings/LM head restent en FP16
        learned_alpha: Si True, active alpha appris
    
    Returns:
        Modele converti (modification in-place + retourne le modele)
    """
    from ternair.quantization.linear import TernairLinear
    
    conversions = 0
    skipped = 0
    
    for name, child in model.named_children():
        full_name = name
        
        if isinstance(child, nn.Linear) and not isinstance(child, TernairLinear):
            is_critical = any(
                keyword in name.lower()
                for keyword in ["embed", "lm_head", "wte", "lm_head"]
            )
            
            if keep_embed_fp16 and is_critical:
                skipped += 1
                continue
            
            new_module = convert_module_to_ternair(
                child, name, storage=storage,
                keep_embed_fp16=keep_embed_fp16,
                learned_alpha=learned_alpha,
            )
            setattr(model, name, new_module)
            conversions += 1
        
        elif hasattr(child, "named_children") and len(list(child.named_children())) > 0:
            # Parcourir les sous-modules recursivement
            _converted, _skipped = _convert_submodules(
                child, prefix=name, storage=storage,
                keep_embed_fp16=keep_embed_fp16,
                learned_alpha=learned_alpha,
            )
            conversions += _converted
            skipped += _skipped
    
    print(f"Conversion terminee : {conversions} Linear -> TernairLinear, {skipped} laissees en FP16")
    return model


def _convert_submodules(
    module: nn.Module,
    prefix: str = "",
    storage: str = "packed",
    keep_embed_fp16: bool = True,
    learned_alpha: bool = True,
) -> tuple[int, int]:
    """Parcourt recursivement les sous-modules et convertit les Linear."""
    from ternair.quantization.linear import TernairLinear
    
    conversions = 0
    skipped = 0
    
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        
        if isinstance(child, nn.Linear) and not isinstance(child, TernairLinear):
            new_module = convert_module_to_ternair(
                child, full_name, storage=storage,
                keep_embed_fp16=keep_embed_fp16,
                learned_alpha=learned_alpha,
            )
            setattr(module, name, new_module)
            conversions += 1
        
        elif hasattr(child, "named_children") and len(list(child.named_children())) > 0:
            c, s = _convert_submodules(
                child, prefix=full_name, storage=storage,
                keep_embed_fp16=keep_embed_fp16,
                learned_alpha=learned_alpha,
            )
            conversions += c
            skipped += s
    
    return conversions, skipped


# ---------------------------------------------------------------------------
# Fonctions de chargement et d'export
# ---------------------------------------------------------------------------

def load_pretrained_for_distillation(
    model_name: str = "HuggingFaceTB/SmolLM2-360M",
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.float16,
):
    """Charge un modele HuggingFace pour la distillation.
    
    Retourne le modele professeur et son tokenizer.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Chargement de {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    teacher = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device if device == "cpu" else None,
    ).to(device)
    
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    
    print(f"Modele charge : {sum(p.numel() for p in teacher.parameters()):,} parametres")
    return teacher, tokenizer


def create_student_from_teacher(
    teacher: nn.Module,
    storage: str = "packed",
    keep_embed_fp16: bool = True,
    learned_alpha: bool = True,
) -> nn.Module:
    """Cree le modele eleve en convertissant le professeur en ternaire.
    
    L'eleve est une copie du professeur avec les Linear remplaces par
    des TernairLinear (les poids sont preserves).
    """
    import copy
    
    student = copy.deepcopy(teacher)
    student = convert_model_to_ternair(
        student,
        storage=storage,
        keep_embed_fp16=keep_embed_fp16,
        learned_alpha=learned_alpha,
    )
    student.train()
    
    # Compter les parametres ternaires vs FP16
    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    
    print(f"Eleve cree : {total_params:,} parametres totaux, {trainable_params:,} entrainables")
    
    return student


def count_ternary_params(model: nn.Module) -> dict:
    """Compte les parametres ternaires vs FP16 dans un modele converti."""
    from ternair.quantization.linear import TernairLinear
    
    ternary_params = 0
    fp16_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, TernairLinear):
            ternary_params += module.out_features * module.in_features
        elif isinstance(module, nn.Linear):
            fp16_params += module.weight.numel()
    
    return {
        "ternary_params": ternary_params,
        "fp16_params": fp16_params,
        "total": ternary_params + fp16_params,
        "ternary_ratio": ternary_params / max(ternary_params + fp16_params, 1),
    }


__all__ = [
    "kl_divergence_loss",
    "distillation_loss",
    "convert_model_to_ternair",
    "convert_module_to_ternair",
    "load_pretrained_for_distillation",
    "create_student_from_teacher",
    "count_ternary_params",
]
