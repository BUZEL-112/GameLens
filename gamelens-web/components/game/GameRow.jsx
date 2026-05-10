import GameCard from "@/components/game/GameCard";
import GameCardSkeleton from "@/components/game/GameCardSkeleton";

/**
 * GameRow — horizontal scroll strip with section title.
 * Uses compact GameCard variant (portrait 3:4 ratio).
 */
export default function GameRow({ title, icon, games, loading, onCardClick }) {
  if (!loading && (!games || games.length === 0)) return null;

  return (
    <section style={{ marginBottom: "2rem" }}>
      {/* Section header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: "1rem",
        }}
      >
        <h2 className="section-title" style={{ fontSize: "1.75rem" }}>
          {icon && (
            <span
              className="material-symbols-outlined"
              style={{ color: "var(--color-primary)", fontSize: "1.75rem" }}
            >
              {icon}
            </span>
          )}
          {title}
        </h2>
        <a
          href="#"
          style={{
            fontSize: "0.65rem",
            fontWeight: 700,
            letterSpacing: "0.06em",
            color: "var(--color-primary)",
            textDecoration: "none",
          }}
        >
          VIEW ALL
        </a>
      </div>

      {/* Scroll strip */}
      <div className="scroll-row">
        {loading
          ? Array.from({ length: 8 }).map((_, i) => (
              <GameCardSkeleton key={i} compact />
            ))
          : games.map((game, idx) => (
              <GameCard
                key={`${game.app_name}-${idx}`}
                game={game}
                compact
                onClick={onCardClick ? () => onCardClick(game) : undefined}
              />
            ))}
      </div>
    </section>
  );
}
