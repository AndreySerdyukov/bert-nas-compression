import { useRef } from "react";

import { N_LAYERS, type LayerMask as Mask } from "../lib/arch";

/**
 * The twelve encoder layers as a strip of cells: filled means kept, hairline means dropped.
 *
 * This is the single visual idea the whole project turns on, so it is one component used
 * everywhere - in the explorer where it is clickable, in the methodology chapters where it labels
 * an architecture, and in the benchmark table where it sits inline at row height. Having one
 * component rather than three variants is what makes a mask in chapter 5 and a mask in the
 * explorer obviously the same kind of thing.
 *
 * Interactive mode is a real radiogroup-style widget rather than twelve independent buttons:
 * arrow keys move between layers, space or enter toggles, and only the focused cell is in the tab
 * order. Twelve tab stops in a row is what makes a keyboard user give up.
 */

export interface LayerMaskProps {
  mask: Mask;
  onToggle?: (index: number) => void;
  /** Cell edge in px. 20 is the inline size used in tables; 34 is the explorer's. */
  size?: number;
  /** Show the layer index inside each cell. Off for the small inline variant. */
  showIndices?: boolean;
  label?: string;
}

export default function LayerMask({
  mask,
  onToggle,
  size = 20,
  showIndices = false,
  label,
}: LayerMaskProps) {
  const interactive = typeof onToggle === "function";
  const cells = useRef<(HTMLButtonElement | null)[]>([]);
  const kept = mask.reduce<number>((total, bit) => total + bit, 0);
  const description = label ?? `${kept} of ${N_LAYERS} encoder layers kept`;

  function move(from: number, delta: number) {
    const next = (from + delta + N_LAYERS) % N_LAYERS;
    cells.current[next]?.focus();
  }

  function onKeyDown(event: React.KeyboardEvent, index: number) {
    if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      event.preventDefault();
      move(index, 1);
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      event.preventDefault();
      move(index, -1);
    } else if (event.key === "Home") {
      event.preventDefault();
      cells.current[0]?.focus();
    } else if (event.key === "End") {
      event.preventDefault();
      cells.current[N_LAYERS - 1]?.focus();
    }
  }

  // The first kept layer, or layer 0, is the one tab reaches - so focus lands somewhere meaningful.
  const tabIndexTarget = Math.max(
    0,
    mask.findIndex((bit) => bit === 1),
  );

  return (
    <div
      className="inline-flex gap-[2px]"
      role={interactive ? "group" : undefined}
      aria-label={interactive ? description : undefined}
    >
      {Array.from({ length: N_LAYERS }, (_, index) => {
        const on = mask[index] === 1;
        const common = {
          className: [
            "flex items-center justify-center rounded-[3px] border transition-colors duration-150",
            on ? "border-accent bg-accent text-canvas" : "border-hair bg-canvas text-slate",
            interactive ? "cursor-pointer hover:border-accent" : "",
          ].join(" "),
          style: { width: size, height: size, fontSize: Math.max(9, size * 0.42) },
        };
        const content = showIndices ? index : "";

        if (!interactive) {
          return (
            <span key={index} {...common} aria-hidden="true">
              {content}
            </span>
          );
        }
        return (
          <button
            key={index}
            type="button"
            ref={(element) => {
              cells.current[index] = element;
            }}
            role="switch"
            aria-checked={on}
            aria-label={`Layer ${index}, ${on ? "kept" : "dropped"}`}
            tabIndex={index === tabIndexTarget ? 0 : -1}
            onClick={() => onToggle?.(index)}
            onKeyDown={(event) => onKeyDown(event, index)}
            {...common}
          >
            {content}
          </button>
        );
      })}
      {!interactive && <span className="sr-only">{description}</span>}
    </div>
  );
}
