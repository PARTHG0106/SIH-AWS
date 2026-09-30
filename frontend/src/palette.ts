// Warm / natural palette, mirrored from app/theme.py and validated for colorblind
// separation (terracotta vs teal). Observed/source is carried by high-contrast ink;
// alerts use a reserved crimson status hue with ring markers.
export type Mode = "light" | "dark";

export interface Palette {
  bgTop: string; bgBottom: string; surface: string; surfaceSoft: string; border: string;
  ink: string; muted: string; faint: string; accent: string; accent2: string; alert: string;
  grid: string; axis: string; areaTop: string; areaBottom: string; shadow: string;
}

export const PALETTES: Record<Mode, Palette> = {
  light: {
    bgTop: "#FBF6EF", bgBottom: "#F1E7D6", surface: "#FFFDF9", surfaceSoft: "#FBF3E7",
    border: "#EADCC6", ink: "#2E2822", muted: "#6E6053", faint: "#9C8B77",
    accent: "#BC5A2E", accent2: "#0E7E70", alert: "#B23A48",
    grid: "#ECE1CF", axis: "#8A7B69",
    areaTop: "rgba(188,90,46,0.20)", areaBottom: "rgba(188,90,46,0.0)",
    shadow: "rgba(122,92,52,0.10)",
  },
  dark: {
    bgTop: "#221E19", bgBottom: "#17130F", surface: "#2A2520", surfaceSoft: "#332C23",
    border: "#3C3429", ink: "#F1E8DA", muted: "#B4A491", faint: "#8A7B67",
    accent: "#E38C5F", accent2: "#48B6A4", alert: "#F0899A",
    grid: "#3A332A", axis: "#9C8E7B",
    areaTop: "rgba(227,140,95,0.26)", areaBottom: "rgba(227,140,95,0.0)",
    shadow: "rgba(0,0,0,0.38)",
  },
};
