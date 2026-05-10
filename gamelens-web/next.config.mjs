/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // Proxy /rec/* requests to the recommendation API
  // In Docker: resolves to http://api:8000
  // Locally: falls back to NEXT_PUBLIC_REC_API_URL or localhost:8000
  async rewrites() {
    const apiBase =
      process.env.REC_API_INTERNAL_URL || "http://api:8000";
    return [
      {
        source: "/rec/:path*",
        destination: `${apiBase}/:path*`,
      },
    ];
  },
};

export default nextConfig;
