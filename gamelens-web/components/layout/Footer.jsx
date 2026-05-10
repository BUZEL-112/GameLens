export default function Footer() {
  return (
    <footer className="footer">
      <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
        <span style={{ color: "#1e293b" }}>
          &copy; {new Date().getFullYear()} GAMELENS — NEXUS DIGITAL ARCHIVE
        </span>
        <div style={{ display: "flex", gap: "1rem" }}>
          <a
            href="#"
            style={{
              color: "#334155",
              textDecoration: "none",
              transition: "color 0.2s",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "#66c0f4")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "#334155")}
          >
            Terms of Service
          </a>
          <a
            href="#"
            style={{
              color: "#334155",
              textDecoration: "none",
              transition: "color 0.2s",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "#66c0f4")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "#334155")}
          >
            Privacy
          </a>
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "2rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
          <div
            style={{
              width: "6px",
              height: "6px",
              borderRadius: "9999px",
              background: "#4ade80",
            }}
          />
          <span style={{ color: "#334155" }}>System Status: Operational</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
          <div
            style={{
              width: "6px",
              height: "6px",
              borderRadius: "9999px",
              background: "#4ade80",
            }}
          />
          <span style={{ color: "#334155" }}>Rec API: Connected</span>
        </div>
      </div>
    </footer>
  );
}
