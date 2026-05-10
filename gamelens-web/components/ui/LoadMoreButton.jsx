export default function LoadMoreButton({ onClick, loading, hasMore, total, loaded }) {
  if (!hasMore) {
    return (
      <div
        style={{
          textAlign: "center",
          padding: "2rem",
          color: "var(--color-outline)",
          fontSize: "0.8rem",
          fontWeight: 600,
          letterSpacing: "0.04em",
        }}
      >
        Showing all {total} games
      </div>
    );
  }

  return (
    <div style={{ textAlign: "center", padding: "2rem 0" }}>
      <button
        onClick={onClick}
        disabled={loading}
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: "0.5rem",
          padding: "0.75rem 2.5rem",
          borderRadius: "0.5rem",
          background: loading
            ? "rgba(186,231,45,0.3)"
            : "var(--color-secondary)",
          color: "var(--color-on-secondary)",
          border: "none",
          fontWeight: 700,
          fontSize: "0.8rem",
          letterSpacing: "0.03em",
          cursor: loading ? "not-allowed" : "pointer",
          transition: "filter 0.15s ease",
        }}
        onMouseEnter={(e) => !loading && (e.currentTarget.style.filter = "brightness(1.1)")}
        onMouseLeave={(e) => (e.currentTarget.style.filter = "none")}
      >
        {loading ? (
          <>
            <span
              className="material-symbols-outlined"
              style={{ fontSize: "1rem", animation: "spin 1s linear infinite" }}
            >
              progress_activity
            </span>
            Loading...
          </>
        ) : (
          <>
            Load 10 more
            <span
              style={{
                opacity: 0.7,
                fontWeight: 400,
                fontSize: "0.75rem",
              }}
            >
              ({total - loaded} remaining)
            </span>
          </>
        )}
      </button>

      <style jsx>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
