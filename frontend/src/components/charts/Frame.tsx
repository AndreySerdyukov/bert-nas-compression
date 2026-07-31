import type { ReactNode } from "react";

import type { Scale } from "./scale";

/**
 * The shared frame every chart in this app draws inside: hairline axes, ticks, labels, and the
 * accessible fallback.
 *
 * The fallback is the part worth explaining. Each chart passes the data it is drawing, and Frame
 * renders it as a visually hidden `<table>` underneath the SVG. That does two jobs at once: a
 * screen reader gets the numbers instead of an unlabelled graphic, and the browser tests assert on
 * table cells rather than on SVG path geometry - so a test breaks when a *value* is wrong, not when
 * a curve moves two pixels.
 */

export interface FrameData {
  columns: string[];
  rows: (string | number)[][];
}

export interface FrameProps {
  width: number;
  height: number;
  /** Room for tick labels. Left is widest because it carries the y-axis text. */
  padding?: { top: number; right: number; bottom: number; left: number };
  x: Scale;
  y: Scale;
  xLabel?: string;
  yLabel?: string;
  xTicks?: number[];
  yTicks?: number[];
  formatX?: (value: number) => string;
  formatY?: (value: number) => string;
  /** Sentence describing what the chart shows, including its headline figures. */
  ariaLabel: string;
  /** The same data, for the screen-reader table and the browser tests. */
  data: FrameData;
  children: ReactNode;
}

const DEFAULT_PADDING = { top: 12, right: 16, bottom: 32, left: 52 };

export default function Frame({
  width,
  height,
  padding = DEFAULT_PADDING,
  x,
  y,
  xLabel,
  yLabel,
  xTicks,
  yTicks,
  formatX = String,
  formatY = String,
  ariaLabel,
  data,
  children,
}: FrameProps) {
  const resolvedXTicks = xTicks ?? x.ticks();
  const resolvedYTicks = yTicks ?? y.ticks();
  const plotBottom = height - padding.bottom;

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full"
        role="img"
        aria-label={ariaLabel}
        // Fonts scale with the viewBox, so sizes are in user units rather than px.
        style={{ overflow: "visible" }}
      >
        {/* Gridlines before the data, so marks always sit on top of them. */}
        {resolvedYTicks.map((tick) => (
          <line
            key={`gy-${tick}`}
            x1={padding.left}
            x2={width - padding.right}
            y1={y(tick)}
            y2={y(tick)}
            stroke="rgb(var(--hair))"
            strokeWidth="1"
            vectorEffect="non-scaling-stroke"
          />
        ))}

        {resolvedYTicks.map((tick) => (
          <text
            key={`ty-${tick}`}
            x={padding.left - 8}
            y={y(tick)}
            textAnchor="end"
            dominantBaseline="middle"
            fill="rgb(var(--slate))"
            fontSize="11"
            style={{ fontVariantNumeric: "tabular-nums" }}
          >
            {formatY(tick)}
          </text>
        ))}

        {resolvedXTicks.map((tick) => (
          <text
            key={`tx-${tick}`}
            x={x(tick)}
            y={plotBottom + 16}
            textAnchor="middle"
            fill="rgb(var(--slate))"
            fontSize="11"
            style={{ fontVariantNumeric: "tabular-nums" }}
          >
            {formatX(tick)}
          </text>
        ))}

        {/* The baseline is drawn last of the frame so it reads as the floor of the plot. */}
        <line
          x1={padding.left}
          x2={width - padding.right}
          y1={plotBottom}
          y2={plotBottom}
          stroke="rgb(var(--slate))"
          strokeWidth="1"
          vectorEffect="non-scaling-stroke"
        />

        {children}
      </svg>

      <div className="mt-1 flex justify-between px-1 text-[11px] text-slate">
        <span>{yLabel}</span>
        <span>{xLabel}</span>
      </div>

      <table className="sr-only">
        <caption>{ariaLabel}</caption>
        <thead>
          <tr>
            {data.columns.map((column) => (
              <th key={column} scope="col">
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.rows.map((row) => (
            <tr key={row.join("|")}>
              {row.map((cell, index) => (
                <td key={`${data.columns[index]}-${cell}`}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}
