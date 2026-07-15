// Présentation : binning du bilan hydrique (mm) en classes nommées réutilisant la
// palette d'anomalie (déficit = rouge = préoccupant). Distinct de la classification
// statistique z→classe qui reste backend (api/era5_anomaly.classify_index).
const BILAN_BANDS: [number, string][] = [
  [-150, 'EXTREMEMENT_BAS'], [-75, 'TRES_BAS'], [-20, 'BAS'],
  [20, 'NORMAL'], [75, 'HAUT'], [150, 'TRES_HAUT'],
]

/**
 * Classifies a climatic water balance value (mm) into one of the 7 McKee-style
 * presentation classes (EXTREMEMENT_BAS…EXTREMEMENT_HAUT). Bands: < -150,
 * [-150,-75), [-75,-20), [-20,20], (20,75], (75,150], > 150.
 * Colours come from SPI_CLASS_COLORS (era5-colors.ts) — deficit = red, surplus = blue.
 */
export function classifyBilan(mm: number): string {
  for (const [hi, cls] of BILAN_BANDS) {
    if (mm < hi) return cls
  }
  return 'EXTREMEMENT_HAUT'
}
