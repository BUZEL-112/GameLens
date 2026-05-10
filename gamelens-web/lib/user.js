/**
 * Client-side anonymous user ID management.
 * Never import from server-side code.
 */
import { v4 as uuidv4 } from "uuid";

const STORAGE_KEY = "gamelens_uid";

/**
 * Get or create a persistent anonymous user ID.
 * Returns null during SSR.
 */
export function getUserId() {
  if (typeof window === "undefined") return null;

  let uid = localStorage.getItem(STORAGE_KEY);
  if (uid) return uid;

  uid = `guest_${uuidv4()}`;
  localStorage.setItem(STORAGE_KEY, uid);
  return uid;
}

/**
 * Determine the base URL for the recommendation API.
 */
function getRecApiBase() {
  // Client-side: if env var set, use it (local dev); otherwise use proxy prefix
  if (process.env.NEXT_PUBLIC_REC_API_URL) {
    return process.env.NEXT_PUBLIC_REC_API_URL;
  }
  return "/rec";
}

/**
 * Fire-and-forget event tracking.
 * Silently catches all errors -- event tracking must never break the UI.
 */
export function recordEvent(userId, itemName, eventType, playtime = 0) {
  if (!userId || !itemName) return;

  const base = getRecApiBase();
  const payload = {
    user_id: userId,
    item_name: itemName,
    event_type: eventType,
    playtime,
    metadata: {},
  };

  fetch(`${base}/v1/events`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }).catch(() => {
    // Silently ignore -- event tracking is best-effort
  });
}
