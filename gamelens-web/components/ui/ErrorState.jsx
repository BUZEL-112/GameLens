export default function ErrorState({ message = "Something went wrong.", onRetry }) {
  return (
    <div
      style={{
        textAlign: "center",
        padding: "4rem 2rem",
        background: "var(--color-surface-container)",
        borderRadius: "0.75rem",
        border: "1px solid rgba(255,180,171,0.2)",
      }}
    >
      <span
        className="material-symbols-outlined"
        style={{
          fontSize: "3rem",
          color: "var(--color-error)",
          display: "block",
          marginBottom: "1rem",
        }}
      >
        error_outline
      </span>
      <p
        style={{
          color: "var(--color-on-surface-variant)",
          fontSize: "0.95rem",
          marginBottom: "1.5rem",
        }}
      >
        {message}
      </p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="btn-glass"
          style={{ display: "inline-flex" }}
        >
          <span className="material-symbols-outlined" style={{ fontSize: "1rem" }}>
            refresh
          </span>
          Try Again
        </button>
      )}
    </div>
  );
}
