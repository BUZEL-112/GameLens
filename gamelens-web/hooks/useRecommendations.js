import { useState, useEffect } from "react";

export function useRecommendations(userId, count = 10) {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!userId) return;

    setLoading(true);
    setError(null);

    const base = process.env.NEXT_PUBLIC_REC_API_URL || "/rec";

    Promise.all([
      fetch(`${base}/v1/recommendations?user_id=${encodeURIComponent(userId)}&count=${count}`)
        .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); }),
      fetch(`/api/games?all=true`)
        .then(r => r.json())
    ])
      .then(([recData, gamesData]) => {
        // Build a lookup map by app_name
        const lookup = {};
        for (const g of gamesData.games || []) lookup[g.app_name] = g;

        // Enrich: match item_name -> full game object
        const enriched = (recData.recommendations || [])
          .map(r => lookup[r.item_name])
          .filter(Boolean);

        setRecommendations(enriched);
        setLoading(false);
      })
      .catch(err => {
        console.error("[useRecommendations]", err.message);
        setError(err.message);
        setLoading(false);
      });
  }, [userId, count]);

  return { recommendations, loading, error };
}