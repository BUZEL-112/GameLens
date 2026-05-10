export default function PriceBadge({ price }) {
  if (price === null || price === undefined) return null;

  const isFree =
    typeof price === "string" &&
    (price.toLowerCase().includes("free"));

  if (isFree) {
    return (
      <span
        style={{
          fontSize: "0.75rem",
          fontWeight: 700,
          color: "#4ade80",
          letterSpacing: "0.01em",
        }}
      >
        Free
      </span>
    );
  }

  return (
    <span
      style={{
        fontSize: "0.8rem",
        fontWeight: 700,
        color: "var(--color-secondary)",
        letterSpacing: "0.01em",
      }}
    >
      {typeof price === "number" ? `$${price.toFixed(2)}` : price}
    </span>
  );
}
