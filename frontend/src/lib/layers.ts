/** The n layers with the highest gate weight, in layer order (1-based). */
export function topLayers(weights: number[], n = 3): number[] {
  return weights
    .map((w, i) => ({ w, layer: i + 1 }))
    .sort((a, b) => b.w - a.w)
    .slice(0, n)
    .map((x) => x.layer)
    .sort((a, b) => a - b)
}
