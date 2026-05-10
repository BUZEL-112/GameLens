import { useState } from "react";
import { sanitize } from "@/lib/sanitize";

/**
 * GameDetailHero — full-width hero section for the game detail page.
 * Matches the Cyber Neon 2077 Store design from Stitch:
 * - Blurred background image with hero vignette
 * - Left: title, genre chips, description, CTA button + price
 * - Right: glassmorphic stats panel (reviews, release date, developer)
 */
export default function GameDetailHero({ game }) {
  const [fallbackLevel, setFallbackLevel] = useState(0);
  const [showInitials, setShowInitials] = useState(false);

  const imageName = sanitize(game.app_name);
  const imageSources = [
    `/images/${imageName}.jpg`,
    game.id
      ? `https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/${game.id}/header.jpg`
      : null,
    "/placeholder.jpg",
  ].filter(Boolean);

  function handleImageError() {
    const next = fallbackLevel + 1;
    if (next < imageSources.length) setFallbackLevel(next);
    else setShowInitials(true);
  }

  const genres = Array.isArray(game.genres) ? game.genres.slice(0, 4) : [];
  const initials = (game.app_name || "??").slice(0, 2).toUpperCase();

  return (
    <section
      style={{
        position: "relative",
        minHeight: "520px",
        overflow: "hidden",
        marginBottom: "3rem",
        display: "flex",
        alignItems: "flex-end",
      }}
    >
      {/* Blurred background */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          zIndex: 0,
        }}
      >
        {showInitials ? (
          <div
            style={{
              width: "100%",
              height: "100%",
              background: "linear-gradient(135deg, #1b2838 0%, #0e1218 100%)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <span
              style={{
                fontSize: "8rem",
                fontWeight: 900,
                color: "rgba(161,217,255,0.05)",
                userSelect: "none",
              }}
            >
              {initials}
            </span>
          </div>
        ) : (
          <img
            src={imageSources[fallbackLevel]}
            alt=""
            onError={handleImageError}
            style={{
              width: "100%",
              height: "100%",
              objectFit: "cover",
              filter: "blur(24px) brightness(0.35)",
              transform: "scale(1.1)",
            }}
          />
        )}
        {/* Vignette gradient */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            background:
              "linear-gradient(to bottom, rgba(19,19,19,0.3) 0%, rgba(19,19,19,0.95) 100%)",
          }}
        />
      </div>

      {/* Content */}
      <div
        style={{
          position: "relative",
          zIndex: 10,
          width: "100%",
          maxWidth: "1200px",
          margin: "0 auto",
          padding: "5rem 2rem 3rem",
          display: "grid",
          gridTemplateColumns: "1fr auto",
          gap: "2rem",
          alignItems: "flex-end",
        }}
      >
        {/* Left: Info */}
        <div>
          {/* Genre chips */}
          {genres.length > 0 && (
            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: "0.5rem",
                marginBottom: "1.25rem",
              }}
            >
              {genres.map((g) => (
                <span key={g} className="genre-chip">
                  {g}
                </span>
              ))}
              {game.sentiment && (
                <span
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    padding: "0.2rem 0.6rem",
                    borderRadius: "9999px",
                    fontSize: "0.65rem",
                    fontWeight: 700,
                    letterSpacing: "0.04em",
                    background: "rgba(186,231,45,0.15)",
                    color: "var(--color-secondary)",
                    border: "1px solid rgba(186,231,45,0.3)",
                    textTransform: "uppercase",
                  }}
                >
                  {game.sentiment}
                </span>
              )}
            </div>
          )}

          {/* Title */}
          <h1
            style={{
              fontSize: "3rem",
              fontWeight: 800,
              lineHeight: 1.1,
              letterSpacing: "-0.02em",
              color: "white",
              marginBottom: "1rem",
              maxWidth: "800px",
            }}
          >
            {game.app_name}
          </h1>

          {/* Specs if available */}
          {Array.isArray(game.specs) && game.specs.length > 0 && (
            <p
              style={{
                fontSize: "1rem",
                color: "var(--color-on-surface-variant)",
                marginBottom: "2rem",
                maxWidth: "600px",
                lineHeight: "1.6",
              }}
            >
              {game.specs.slice(0, 4).join(" · ")}
            </p>
          )}

          {/* CTA */}
          <div style={{ display: "flex", alignItems: "center", gap: "1.5rem" }}>
            <button className="btn-primary">
              <span className="material-symbols-outlined" style={{ fontSize: "1.1rem" }}>
                library_add
              </span>
              Add to Library
            </button>
            <button className="btn-glass">View Details</button>

            {/* Price */}
            <div>
              {game.price === "Free" || game.price === "Free to Play" ? (
                <span
                  style={{
                    fontSize: "1.5rem",
                    fontWeight: 700,
                    color: "#4ade80",
                  }}
                >
                  Free
                </span>
              ) : game.price !== null && game.price !== undefined ? (
                <span
                  style={{
                    fontSize: "1.5rem",
                    fontWeight: 700,
                    color: "var(--color-secondary)",
                  }}
                >
                  {typeof game.price === "number"
                    ? `$${game.price.toFixed(2)}`
                    : game.price}
                </span>
              ) : null}
            </div>
          </div>
        </div>

        {/* Right: Stats panel (glassmorphic) */}
        <div
          className="glass"
          style={{
            padding: "1.5rem",
            borderRadius: "0.75rem",
            minWidth: "220px",
            display: "flex",
            flexDirection: "column",
            gap: "0",
          }}
        >
          {game.sentiment && (
            <StatRow label="RECENT REVIEWS" value={game.sentiment} highlight />
          )}
          {game.release_date && (
            <StatRow label="RELEASE DATE" value={game.release_date} />
          )}
          {game.developer && (
            <StatRow label="DEVELOPER" value={game.developer} link />
          )}
          {game.publisher && game.publisher !== game.developer && (
            <StatRow label="PUBLISHER" value={game.publisher} />
          )}
        </div>
      </div>
    </section>
  );
}

function StatRow({ label, value, highlight, link }) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "1rem 0",
        borderBottom: "1px solid rgba(255,255,255,0.05)",
        gap: "1rem",
      }}
    >
      <span
        style={{
          fontSize: "0.65rem",
          fontWeight: 700,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          color: "var(--color-outline)",
        }}
      >
        {label}
      </span>
      <span
        style={{
          fontSize: "0.8rem",
          fontWeight: highlight ? 700 : 400,
          color: highlight
            ? "var(--color-primary)"
            : link
            ? "var(--color-primary)"
            : "var(--color-on-surface)",
          textAlign: "right",
          maxWidth: "120px",
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
          cursor: link ? "pointer" : "default",
        }}
      >
        {value}
      </span>
    </div>
  );
}
