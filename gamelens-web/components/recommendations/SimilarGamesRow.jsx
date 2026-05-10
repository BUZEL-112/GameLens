import { useRouter } from "next/router";
import { useSimilarGames } from "@/hooks/useSimilarGames";
import GameRow from "@/components/game/GameRow";

/**
 * SimilarGamesRow — "More Like This" section on the game detail page.
 * Silently renders nothing if rec API is down or returns empty results.
 */
export default function SimilarGamesRow({ appName }) {
  const router = useRouter();
  const { similar, loading } = useSimilarGames(appName);

  function handleCardClick(game) {
    router.push(`/game/${encodeURIComponent(game.app_name)}`);
  }

  if (!loading && similar.length === 0) return null;

  return (
    <GameRow
      title="More Like This"
      icon="recommend"
      games={similar}
      loading={loading}
      onCardClick={handleCardClick}
    />
  );
}
