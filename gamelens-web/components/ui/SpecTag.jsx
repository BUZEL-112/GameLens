export default function SpecTag({ label }) {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "0.25rem",
        padding: "0.25rem 0.6rem",
        borderRadius: "0.375rem",
        fontSize: "0.7rem",
        fontWeight: 600,
        letterSpacing: "0.03em",
        background: "rgba(27, 40, 56, 0.8)",
        color: "var(--color-primary)",
        border: "1px solid rgba(255,255,255,0.08)",
      }}
    >
      <span className="material-symbols-outlined" style={{ fontSize: "0.8rem" }}>
        check_circle
      </span>
      {label}
    </span>
  );
}
