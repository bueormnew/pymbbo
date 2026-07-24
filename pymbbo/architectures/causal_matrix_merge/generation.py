"""Utilidades de muestreo para generación autoregresiva.

Provee funciones standalone para escalar logits por temperatura,
filtrar por top-k, aplicar muestreo nucleus (top-p) y un pipeline
completo de muestreo de tokens.
"""

import torch
from typing import Optional


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Escala logits por temperatura.

    Divide los logits por el factor de temperatura para controlar la
    aleatoriedad del muestreo. Temperaturas mayores a 1.0 producen
    distribuciones más uniformes (más aleatorias), mientras que
    temperaturas menores a 1.0 concentran la masa de probabilidad
    en los tokens más probables.

    Args:
        logits: Tensor de forma [batch, vocab_size] con logits crudos.
        temperature: Factor de escala. Debe ser estrictamente mayor que 0.

    Returns:
        Tensor de forma [batch, vocab_size] con logits escalados
        (logits / temperature).

    Raises:
        ValueError: Si temperature <= 0.
    """
    if temperature <= 0:
        raise ValueError("temperature debe ser > 0")
    return logits / temperature


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Filtra logits manteniendo solo los top-k valores más altos.

    Los valores fuera de las k posiciones más altas se establecen a
    -inf, de modo que tras aplicar softmax tendrán probabilidad cero.

    Args:
        logits: Tensor de forma [batch, vocab_size] con logits.
        k: Número de tokens a mantener. Debe ser >= 1.

    Returns:
        Tensor de forma [batch, vocab_size] donde a lo sumo k
        posiciones por fila tienen valores finitos y el resto es -inf.

    Raises:
        ValueError: Si k < 1.
    """
    if k < 1:
        raise ValueError("top_k debe ser >= 1")

    # Si k >= vocab_size, no hay nada que filtrar
    if k >= logits.size(-1):
        return logits

    # Obtener el valor del k-ésimo logit más alto por fila
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    # El umbral es el valor mínimo entre los top-k
    threshold = top_k_values[..., -1].unsqueeze(-1)

    # Poner -inf donde el logit es menor que el umbral
    mask = logits < threshold
    return logits.masked_fill(mask, float("-inf"))


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Aplica muestreo nucleus (top-p).

    Mantiene el conjunto mínimo de tokens cuya probabilidad acumulada
    supera p. Los logits se ordenan de mayor a menor, se computa la
    probabilidad acumulada con softmax, y se descartan (poniéndolos a
    -inf) los tokens cuya inclusión excede el umbral p.

    Args:
        logits: Tensor de forma [batch, vocab_size] con logits.
        p: Umbral de probabilidad acumulada. Debe estar en (0, 1].

    Returns:
        Tensor de forma [batch, vocab_size] con logits filtrados por
        nucleus sampling.

    Raises:
        ValueError: Si p <= 0 o p > 1.
    """
    if p <= 0 or p > 1:
        raise ValueError("top_p debe estar en (0, 1]")

    # Si p == 1.0, mantener todo
    if p == 1.0:
        return logits

    # Ordenar logits en orden descendente
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)

    # Calcular probabilidades acumuladas
    cumulative_probs = torch.cumsum(
        torch.softmax(sorted_logits, dim=-1), dim=-1
    )

    # Crear máscara: descartar tokens cuya probabilidad acumulada excede p
    # Desplazamos la máscara una posición a la derecha para incluir el primer
    # token que cruza el umbral
    sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= p

    # Aplicar máscara a los logits ordenados
    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

    # Restaurar el orden original
    output = torch.zeros_like(logits)
    output.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    return output


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """Pipeline completo de muestreo: temperature → top_k → top_p → multinomial.

    Aplica secuencialmente los filtros de temperatura, top-k y top-p
    (según se proporcionen), y luego muestrea un token de la distribución
    resultante usando muestreo multinomial.

    Si tras aplicar los filtros todos los logits son -inf (caso extremo),
    se recurre a argmax sobre los logits originales como fallback.

    Args:
        logits: Tensor de forma [batch, vocab_size] con logits crudos.
        temperature: Factor de escala de temperatura. Default: 1.0.
        top_k: Si se proporciona, filtra a los top-k logits más altos.
        top_p: Si se proporciona, aplica muestreo nucleus con umbral p.

    Returns:
        Tensor de forma [batch, 1] con los token IDs muestreados.

    Raises:
        ValueError: Si temperature <= 0, top_k < 1, o top_p fuera de (0, 1].
    """
    # Guardar logits originales para fallback
    original_logits = logits

    # 1. Aplicar temperatura
    logits = apply_temperature(logits, temperature)

    # 2. Aplicar top-k si se proporciona
    if top_k is not None:
        logits = apply_top_k(logits, top_k)

    # 3. Aplicar top-p si se proporciona
    if top_p is not None:
        logits = apply_top_p(logits, top_p)

    # 4. Manejar caso extremo: todos los logits son -inf
    all_neg_inf = torch.all(logits == float("-inf"), dim=-1)
    if all_neg_inf.any():
        # Fallback a argmax para las filas con todos -inf
        fallback_tokens = original_logits.argmax(dim=-1, keepdim=True)
        # Para las filas normales, proceder con multinomial
        probs = torch.softmax(logits, dim=-1)
        # Reemplazar NaN/inf en probs (de filas con todos -inf) con uniforme
        # para evitar errores en multinomial
        safe_probs = probs.clone()
        safe_probs[all_neg_inf] = 1.0 / logits.size(-1)
        sampled_tokens = torch.multinomial(safe_probs, num_samples=1)
        # Usar fallback donde corresponde
        result = torch.where(
            all_neg_inf.unsqueeze(-1), fallback_tokens, sampled_tokens
        )
        return result

    # Caso normal: muestreo multinomial
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
