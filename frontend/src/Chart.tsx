import { useEffect, useRef } from "react";
import * as echarts from "echarts";
import type { Palette } from "./palette";
import type { UsaChannel, InChannel } from "./api";

export function EChart({ option, height = 250 }: { option: echarts.EChartsCoreOption; height?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts>();
  useEffect(() => {
    if (!ref.current) return;
    chart.current = echarts.init(ref.current, undefined, { renderer: "canvas" });
    const ro = new ResizeObserver(() => chart.current?.resize());
    ro.observe(ref.current);
    return () => { ro.disconnect(); chart.current?.dispose(); };
  }, []);
  useEffect(() => { chart.current?.setOption(option, true); }, [option]);
  return <div ref={ref} style={{ width: "100%", height }} />;
}

const AXIS = (p: Palette) => ({
  axisLabel: { color: p.axis, fontFamily: "Inter", fontSize: 11 },
  axisLine: { lineStyle: { color: p.grid } },
  axisTick: { lineStyle: { color: p.grid } },
});

function tooltip(p: Palette, trigger: "axis" | "item") {
  return {
    trigger, backgroundColor: p.surface, borderColor: p.border, borderWidth: 1,
    textStyle: { color: p.ink, fontFamily: "Inter", fontSize: 12 },
    axisPointer: { type: "line", lineStyle: { color: p.faint, width: 1 } },
    extraCssText: "border-radius:10px; box-shadow:0 4px 16px rgba(0,0,0,0.12);",
  };
}

function legend(p: Palette, data: string[]) {
  return { top: 2, right: 6, icon: "roundRect", itemWidth: 14, itemHeight: 4,
    textStyle: { color: p.muted, fontFamily: "Inter", fontSize: 12 }, data };
}

function zoom(p: Palette) {
  return [
    { type: "inside", throttle: 40 },
    { type: "slider", height: 16, bottom: 8, borderColor: p.border, backgroundColor: "transparent",
      fillerColor: p.areaTop, handleStyle: { color: p.accent },
      dataBackground: { lineStyle: { color: p.faint }, areaStyle: { color: p.grid } },
      textStyle: { color: p.axis, fontSize: 10 } },
  ];
}
// __CHART_MORE__


export function timeSeriesOption(ch: UsaChannel, p: Palette, show60: boolean): echarts.EChartsCoreOption {
  const series: any[] = [
    { name: "Observed", type: "line", data: ch.observed, showSymbol: false, connectNulls: false, sampling: "lttb",
      lineStyle: { width: 2, color: p.ink }, itemStyle: { color: p.ink }, z: 5,
      areaStyle: { opacity: 1, color: new echarts.graphic.LinearGradient(0, 0, 0, 1,
        [{ offset: 0, color: p.areaTop }, { offset: 1, color: p.areaBottom }]) } },
    { name: "Model · 1-min", type: "line", data: ch.model_1m, showSymbol: false, connectNulls: false, sampling: "lttb",
      lineStyle: { width: 1.6, color: p.accent, type: "dashed" }, itemStyle: { color: p.accent }, z: 4 },
  ];
  if (show60)
    series.push({ name: "Model · 60-min", type: "line", data: ch.model_60m, showSymbol: false, connectNulls: false,
      sampling: "lttb", lineStyle: { width: 1.6, color: p.accent2, type: [2, 3] }, itemStyle: { color: p.accent2 }, z: 4 });
  series.push({ name: "Candidate", type: "scatter", data: ch.candidates.map((c) => [c[0], c[1]]), symbol: "circle",
    symbolSize: 9, itemStyle: { color: "transparent", borderColor: p.alert, borderWidth: 1.8 }, z: 6 });
  const names = ["Observed", "Model · 1-min", ...(show60 ? ["Model · 60-min"] : []), "Candidate"];
  return {
    grid: { left: 54, right: 16, top: 42, bottom: 58 },
    tooltip: tooltip(p, "axis"), legend: legend(p, names),
    xAxis: { type: "time", ...AXIS(p), splitLine: { show: false } },
    yAxis: { type: "value", scale: true, name: ch.unit, nameGap: 12,
      nameTextStyle: { color: p.muted, fontFamily: "Inter", fontSize: 11, align: "left" },
      ...AXIS(p), axisLine: { show: false }, splitLine: { lineStyle: { color: p.grid, type: "dashed" } } },
    dataZoom: zoom(p), series, textStyle: { fontFamily: "Inter" }, animationDuration: 400,
  };
}

export function heatmapOption(matrix: number[][], labels: string[], p: Palette): echarts.EChartsCoreOption {
  const rowSums = matrix.map((r) => r.reduce((a, b) => a + b, 0) || 1);
  const data: any[] = [];
  let max = 0;
  matrix.forEach((row, i) => row.forEach((v, j) => {
    const frac = v / rowSums[i];
    max = Math.max(max, frac);
    data.push([j, i, frac, v]);
  }));
  return {
    grid: { left: 120, right: 20, top: 20, bottom: 96 },
    tooltip: { ...tooltip(p, "item"),
      formatter: (o: any) => `true <b>${labels[o.value[1]]}</b> → pred <b>${labels[o.value[0]]}</b><br/>${o.value[3]} rows (${(o.value[2] * 100).toFixed(0)}%)` },
    xAxis: { type: "category", data: labels, axisLabel: { color: p.axis, rotate: 45, fontSize: 10 },
      axisLine: { lineStyle: { color: p.grid } }, splitArea: { show: false } },
    yAxis: { type: "category", data: labels, axisLabel: { color: p.axis, fontSize: 10 },
      axisLine: { lineStyle: { color: p.grid } } },
    visualMap: { min: 0, max: max || 1, show: false, inRange: { color: [p.surface, p.accent] } },
    series: [{ type: "heatmap", data, label: { show: true, color: p.ink, fontSize: 9,
      formatter: (o: any) => (o.value[2] > 0.04 ? (o.value[2] * 100).toFixed(0) : "") },
      itemStyle: { borderColor: p.surface, borderWidth: 1 } }],
    textStyle: { fontFamily: "Inter" },
  };
}

export function barOption(items: { name: string; value: number }[], p: Palette): echarts.EChartsCoreOption {
  const rows = items.slice().reverse();
  return {
    grid: { left: 190, right: 24, top: 8, bottom: 24 },
    tooltip: { ...tooltip(p, "item"), formatter: (o: any) => `${o.name}: ${o.value.toFixed(3)}` },
    xAxis: { type: "value", ...AXIS(p), splitLine: { lineStyle: { color: p.grid, type: "dashed" } } },
    yAxis: { type: "category", data: rows.map((r) => r.name),
      axisLabel: { color: p.axis, fontSize: 10 }, axisLine: { lineStyle: { color: p.grid } } },
    series: [{ type: "bar", data: rows.map((r) => r.value), itemStyle: { color: p.accent2, borderRadius: [0, 3, 3, 0] }, barWidth: "62%" }],
    textStyle: { fontFamily: "Inter" },
  };
}

export function scatterOption(ch: InChannel, p: Palette): echarts.EChartsCoreOption {
  const mk = (data: any[], color: string, symbol: string, size: number) => ({
    type: "scatter", data: data.map((d) => [d[0], d[1]]), symbol, symbolSize: size,
    itemStyle: { color, borderColor: color, opacity: 0.9 },
  });
  return {
    grid: { left: 54, right: 16, top: 42, bottom: 58 },
    tooltip: tooltip(p, "axis"),
    legend: legend(p, [ch.source_label, "Synthetic scenario copy", "Applied scenario"]),
    xAxis: { type: "time", ...AXIS(p), splitLine: { show: false } },
    yAxis: { type: "value", scale: true, ...AXIS(p), axisLine: { show: false },
      splitLine: { lineStyle: { color: p.grid, type: "dashed" } } },
    dataZoom: zoom(p),
    series: [
      { name: ch.source_label, ...mk(ch.source, p.ink, "circle", 8) },
      { name: "Synthetic scenario copy", ...mk(ch.synthetic, p.accent, "triangle", 9) },
      { name: "Applied scenario", ...mk(ch.applied, p.alert, "diamond", 13) },
    ],
    textStyle: { fontFamily: "Inter" }, animationDuration: 300,
  };
}
