import { useContext } from "react";
import { useRouter } from "next/router";
import { UserContext } from "@/pages/_app";
import { useRecommendations } from "@/hooks/useRecommendations";
import GameRow from "@/components/game/GameRow";

/**
 * PersonalizedRow — "Recommended For You" bento section on the homepage.
 * Silently fails if the rec API is unreachable.
 */
export default function PersonalizedRow() {
  const userId = useContext(UserContext);
  const router = useRouter();
  const { recommendations, loading } = useRecommendations(userId, 12);

  function handleCardClick(game) {
    router.push(`/game/${encodeURIComponent(game.app_name)}`);
  }

  const title = userId ? "Recommended For You" : "Popular Right Now";
  const icon = "auto_awesome";

  // If not loading and no recs, render nothing (silent failure)
  if (!loading && recommendations.length === 0) return null;

  return (
    <GameRow
      title={title}
      icon={icon}
      games={recommendations}
      loading={loading}
      onCardClick={handleCardClick}
    />
  );
}
