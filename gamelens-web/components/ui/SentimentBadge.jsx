const SENTIMENT_COLORS = {
  "Overwhelmingly Positive": { bg: "rgba(74,222,128,0.12)", color: "#4ade80", border: "rgba(74,222,128,0.25)" },
  "Very Positive": { bg: "rgba(74,222,128,0.1)", color: "#86efac", border: "rgba(74,222,128,0.2)" },
  Positive: { bg: "rgba(161,217,255,0.1)", color: "#a1d9ff", border: "rgba(161,217,255,0.2)" },
  "Mostly Positive": { bg: "rgba(161,217,255,0.08)", color: "#93c5fd", border: "rgba(161,217,255,0.15)" },
  Mixed: { bg: "rgba(251,191,36,0.1)", color: "#fbbf24", border: "rgba(251,191,36,0.2)" },
  "Mostly Negative": { bg: "rgba(248,113,113,0.1)", color: "#f87171", border: "rgba(248,113,113,0.2)" },
  Negative: { bg: "rgba(248,113,113,0.12)", color: "#ef4444", border: "rgba(248,113,113,0.25)" },
};

export default function SentimentBadge({ sentiment }) {
  if (!sentiment) return null;

  const style = SENTIMENT_COLORS[sentiment] || {
    bg: "rgba(255,255,255,0.05)",
    color: "var(--color-outline)",
    border: "rgba(255,255,255,0.1)",
  };

  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        padding: "0.15rem 0.5rem",
        borderRadius: "0.25rem",
        fontSize: "0.6rem",
        fontWeight: 700,
        letterSpacing: "0.03em",
        background: style.bg,
        color: style.color,
        border: `1px solid ${style.border}`,
        whiteSpace: "nowrap",
      }}
    >
      {sentiment}
    </span>
  );
}
