import { useState, useEffect } from "react";

/**
 * Fetch personalized recommendations for a user.
 * Returns empty state if userId is null (SSR or not yet initialized).
 */
export function useRecommendations(userId, count = 10) {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!userId) return;

    setLoading(true);
    setError(null);

    const base = process.env.NEXT_PUBLIC_REC_API_URL || "/rec";
    const url = `${base}/v1/recommendations?user_id=${encodeURIComponent(userId)}&count=${count}`;

    fetch(url)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        setRecommendations(data.recommendations || []);
        setLoading(false);
      })
      .catch((err) => {
        console.error("[useRecommendations]", err.message);
        setError(err.message);
        setLoading(false);
      });
  }, [userId, count]);

  return { recommendations, loading, error };
}
