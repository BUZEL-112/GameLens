/**
 * PageWrapper — constrains main content to the right of the sidebar.
 * No additional max-width needed here; pages control their own layout.
 */
export default function PageWrapper({ children, className = "" }) {
  return (
    <main className={`main-content page-enter ${className}`}>
      {children}
    </main>
  );
}
