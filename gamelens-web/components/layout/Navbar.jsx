import { useState, useContext } from "react";
import { useRouter } from "next/router";
import Link from "next/link";
import { UserContext } from "@/pages/_app";

export default function Navbar() {
  const router = useRouter();
  const userId = useContext(UserContext);
  const [searchValue, setSearchValue] = useState("");

  function handleSearch(e) {
    e.preventDefault();
    const q = searchValue.trim();
    if (q) {
      router.push(`/search?q=${encodeURIComponent(q)}`);
    }
  }

  const truncatedId = userId
    ? userId.length > 12
      ? userId.slice(0, 12) + "..."
      : userId
    : null;

  return (
    <header className="navbar">
      {/* Left: Logo + Nav links */}
      <div style={{ display: "flex", alignItems: "center", gap: "2rem" }}>
        <Link href="/" style={{ textDecoration: "none" }}>
          <span
            style={{
              fontSize: "1.5rem",
              fontWeight: 900,
              fontStyle: "italic",
              letterSpacing: "-0.04em",
              color: "white",
            }}
          >
            GAMELENS
          </span>
        </Link>

        <nav
          style={{
            display: "flex",
            gap: "1.5rem",
            alignItems: "center",
          }}
          className="hide-mobile"
        >
          <NavLink href="/" label="Store" active={router.pathname === "/"} />
          <NavLink
            href="/search"
            label="Browse"
            active={router.pathname === "/search"}
          />
        </nav>
      </div>

      {/* Center: Search */}
      <form
        onSubmit={handleSearch}
        style={{ position: "relative", flex: "0 1 320px" }}
        className="hide-mobile"
      >
        <span
          className="material-symbols-outlined"
          style={{
            position: "absolute",
            left: "0.75rem",
            top: "50%",
            transform: "translateY(-50%)",
            fontSize: "1.1rem",
            color: "#64748b",
            pointerEvents: "none",
          }}
        >
          search
        </span>
        <input
          type="text"
          value={searchValue}
          onChange={(e) => setSearchValue(e.target.value)}
          placeholder="Search the archive..."
          className="search-input"
        />
      </form>

      {/* Right: Icons + User */}
      <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
        <button
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            color: "#64748b",
            display: "flex",
            alignItems: "center",
            transition: "color 0.2s",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.color = "white")}
          onMouseLeave={(e) => (e.currentTarget.style.color = "#64748b")}
          aria-label="Notifications"
        >
          <span className="material-symbols-outlined">notifications</span>
        </button>

        <button
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            color: "#64748b",
            display: "flex",
            alignItems: "center",
            transition: "color 0.2s",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.color = "white")}
          onMouseLeave={(e) => (e.currentTarget.style.color = "#64748b")}
          aria-label="Settings"
        >
          <span className="material-symbols-outlined">settings</span>
        </button>

        {/* User avatar / ID */}
        <div suppressHydrationWarning>
          {truncatedId ? (
            <Link
              href={`/user/${encodeURIComponent(userId)}`}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.5rem",
                textDecoration: "none",
              }}
            >
              <div
                style={{
                  width: "2.25rem",
                  height: "2.25rem",
                  borderRadius: "9999px",
                  background:
                    "linear-gradient(135deg, #66c0f4 0%, #bae72d 100%)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: "0.75rem",
                  fontWeight: 800,
                  color: "#001e2d",
                  border: "2px solid rgba(102, 192, 244, 0.3)",
                  flexShrink: 0,
                }}
              >
                {truncatedId.slice(6, 8).toUpperCase()}
              </div>
            </Link>
          ) : (
            <div
              style={{
                width: "2.25rem",
                height: "2.25rem",
                borderRadius: "9999px",
                background: "rgba(255,255,255,0.05)",
                border: "1px solid rgba(255,255,255,0.1)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <span
                className="material-symbols-outlined"
                style={{ fontSize: "1.2rem", color: "#64748b" }}
              >
                person
              </span>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}

function NavLink({ href, label, active }) {
  return (
    <Link
      href={href}
      style={{
        textDecoration: "none",
        color: active ? "#66c0f4" : "#94a3b8",
        fontWeight: 500,
        fontSize: "0.9rem",
        paddingBottom: active ? "0.25rem" : "0",
        borderBottom: active ? "2px solid #66c0f4" : "2px solid transparent",
        transition: "all 0.2s ease",
      }}
    >
      {label}
    </Link>
  );
}
