import { createContext, useState, useEffect } from "react";
import { useRouter } from "next/router";
import { getUserId } from "@/lib/user";
import Navbar from "@/components/layout/Navbar";
import Sidebar from "@/components/layout/Sidebar";
import Footer from "@/components/layout/Footer";
import "@/styles/globals.css";

export const UserContext = createContext(null);

export default function App({ Component, pageProps }) {
  const [userId, setUserId] = useState(null);
  const [selectedGenre, setSelectedGenre] = useState(null);
  const router = useRouter();

  useEffect(() => {
    const id = getUserId();
    setUserId(id);
  }, []);

  // Sync selectedGenre from URL query for Sidebar highlighting
  useEffect(() => {
    const g = router.query.genre || null;
    setSelectedGenre(g || null);
  }, [router.query.genre]);

  return (
    <UserContext.Provider value={userId}>
      <Navbar />
      <Sidebar selectedGenre={selectedGenre} />
      <Component {...pageProps} key={router.asPath} />
      <Footer />
    </UserContext.Provider>
  );
}
