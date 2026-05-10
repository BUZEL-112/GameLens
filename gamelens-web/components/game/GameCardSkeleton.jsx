/**
 * GameCardSkeleton — shimmering placeholder matching GameCard dimensions.
 */
export default function GameCardSkeleton({ compact = false }) {
  return (
    <div
      style={{
        borderRadius: "0.75rem",
        overflow: "hidden",
        background: "var(--color-surface-container)",
        border: "1px solid rgba(255,255,255,0.05)",
      }}
    >
      {/* Image placeholder */}
      <div
        className="skeleton"
        style={{
          width: "100%",
          paddingBottom: compact ? "133%" : "46.7%",
        }}
      />
      {/* Content placeholders */}
      <div style={{ padding: "0.75rem", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
        <div
          className="skeleton"
          style={{ height: "0.85rem", borderRadius: "0.25rem", width: "75%" }}
        />
        <div
          className="skeleton"
          style={{ height: "0.7rem", borderRadius: "0.25rem", width: "50%" }}
        />
        <div
          className="skeleton"
          style={{ height: "0.7rem", borderRadius: "0.25rem", width: "30%" }}
        />
      </div>
    </div>
  );
}
