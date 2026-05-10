import { useRouter } from "next/router";
import { useGames } from "@/hooks/useGames";
import GameCard from "@/components/game/GameCard";
import GameCardSkeleton from "@/components/game/GameCardSkeleton";
import LoadMoreButton from "@/components/ui/LoadMoreButton";
import ErrorState from "@/components/ui/ErrorState";

export default function GameGrid({ genre, q }) {
  const router = useRouter();
  const { games, total, hasMore, loading, loadingMore, error, loadMore, reload } =
    useGames(genre, q);

  function handleCardClick(game) {
    router.push(`/game/${encodeURIComponent(game.app_name)}`);
  }

  if (loading) {
    return (
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
          gap: "1.5rem",
        }}
      >
        {Array.from({ length: 10 }).map((_, i) => (
          <GameCardSkeleton key={i} />
        ))}
      </div>
    );
  }

  if (error) {
    return <ErrorState message="Failed to load games." onRetry={reload} />;
  }

  if (games.length === 0) {
    return (
      <div
        style={{
          textAlign: "center",
          padding: "4rem 2rem",
          color: "var(--color-outline)",
        }}
      >
        <span
          className="material-symbols-outlined"
          style={{ fontSize: "3rem", display: "block", marginBottom: "1rem", color: "var(--color-outline-variant)" }}
        >
          search_off
        </span>
        <p style={{ fontSize: "1rem", fontWeight: 600, color: "var(--color-on-surface-variant)" }}>
          No games found
        </p>
        <p style={{ fontSize: "0.85rem", color: "var(--color-outline)", marginTop: "0.5rem" }}>
          Try clearing your filters or searching for something else
        </p>
      </div>
    );
  }

  return (
    <>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
          gap: "1.5rem",
          marginBottom: "2rem",
        }}
      >
        {games.map((game, idx) => (
          <GameCard
            key={`${game.app_name}-${idx}`}
            game={game}
            onClick={handleCardClick}
          />
        ))}
      </div>

      <LoadMoreButton
        onClick={loadMore}
        loading={loadingMore}
        hasMore={hasMore}
        total={total}
        loaded={games.length}
      />
    </>
  );
}
