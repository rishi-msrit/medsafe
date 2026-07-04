import { Link, Outlet, useLocation } from 'react-router-dom'
import { useState, useEffect } from 'react'
import { Menu, X } from 'lucide-react'

const navItems = [
  { to: '/',              label: 'Analyzer',     id: 'nav-analyzer' },
  { to: '/pairwise',     label: 'Pairwise',     id: 'nav-pairwise' },
  { to: '/alternatives', label: 'Alternatives', id: 'nav-alternatives' },
  { to: '/molecular',    label: 'Explore',      id: 'nav-explore' },
  { to: '/directory',    label: 'Directory',    id: 'nav-directory' },
  { to: '/neighborhood', label: 'Neighborhood', id: 'nav-neighborhood' },
  { to: '/about',        label: 'About',        id: 'nav-about' },
]


export function Layout() {
  const location = useLocation()
  const [mobileOpen, setMobileOpen] = useState(false)

  // Initialize theme from storage or preference
  const [isDark, setIsDark] = useState(() => {
    const cached = localStorage.getItem('theme')
    if (cached) return cached === 'dark'
    return window.matchMedia('(prefers-color-scheme: dark)').matches
  })

  // Apply dark mode class
  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark-mode')
      localStorage.setItem('theme', 'dark')
    } else {
      document.documentElement.classList.remove('dark-mode')
      localStorage.setItem('theme', 'light')
    }
  }, [isDark])

  const isActive = (to: string) =>
    to === '/' ? location.pathname === '/' : location.pathname.startsWith(to)

  return (
    <div className="min-h-screen flex flex-col bg-[var(--bg-app)] text-[var(--text-primary)] transition-colors duration-200">
      {/* Top Header Nav - Pure White, Bottom Border, No Shadow */}
      <nav className="sticky top-0 z-50 bg-[#FFFFFF] dark:bg-[#1C1C1E] border-b border-[#E2DED8] dark:border-[#2C2C2E]">
        <div className="max-w-[1280px] mx-auto px-6">
          <div className="flex items-center justify-between h-14">
            {/* Brand/Logo - Shield Cross Icon + DM Serif Display */}
            <Link 
              to="/" 
              id="nav-logo" 
              className="flex items-center gap-2 font-normal text-[20px] text-[#111111] dark:text-[#FAFAFA] hover:opacity-85 transition-opacity no-underline serif-heading"
            >
              <svg className="w-5 h-5 stroke-current" viewBox="0 0 24 24" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                <path d="M12 8v8M9 12h6"/>
              </svg>
              <span>MedSafe</span>
            </Link>

            {/* Header Controls */}
            <div className="flex items-center gap-5">
              {/* Desktop Navigation Links */}
              <div className="hidden md:flex items-center gap-6">
                {navItems.map(({ to, label, id }) => (
                  <Link
                    key={to}
                    to={to}
                    id={id}
                    className={`text-[14px] no-underline transition-colors duration-150 py-1.5 border-b-2 ${
                      isActive(to)
                        ? 'text-[#111111] dark:text-[#FAFAFA] font-medium border-b-[#111111] dark:border-b-[#FAFAFA]'
                        : 'text-[#6B7280] dark:text-[#9CA3AF] hover:text-[#111111] dark:hover:text-[#FAFAFA] border-b-transparent'
                    }`}
                  >
                    {label}
                  </Link>
                ))}
              </div>


              {/* Mobile Hamburger menu button */}
              <button
                onClick={() => setMobileOpen(!mobileOpen)}
                className="md:hidden p-1 text-[var(--text-primary)] bg-transparent border-0 cursor-pointer flex items-center"
                aria-label="Toggle navigation menu"
              >
                {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </button>
            </div>
          </div>
        </div>

        {/* Mobile Dropdown Menu */}
        {mobileOpen && (
          <div className="md:hidden bg-[#FFFFFF] dark:bg-[#1C1C1E] border-t border-[#E2DED8] dark:border-[#2C2C2E] px-6 py-3 flex flex-col gap-2.5 animate-fade-in">
            {navItems.map(({ to, label, id }) => (
              <Link
                key={to}
                to={to}
                id={`${id}-mobile`}
                onClick={() => setMobileOpen(false)}
                className={`text-[14px] no-underline py-2 block transition-colors ${
                  isActive(to)
                    ? 'text-[#111111] dark:text-[#FAFAFA] font-medium border-l-2 border-l-[#111111] dark:border-l-[#FAFAFA] pl-2'
                    : 'text-[#6B7280] dark:text-[#9CA3AF] hover:text-[#111111] pl-2'
                }`}
              >
                {label}
              </Link>
            ))}
          </div>
        )}
      </nav>

      {/* Main Content Area - Styled with Dot Grid Background */}
      <main className="content-dot-grid flex-1 w-full max-w-[1280px] mx-auto px-6 py-12 flex flex-col gap-8 relative">
        <Outlet />
      </main>

      {/* Global Clinical Bottom Disclaimer */}
      <footer className="w-full max-w-[1280px] mx-auto px-6 py-8 mt-auto text-center border-t border-[#E2DED8] dark:border-[#2C2C2E]">
        <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed m-0 font-normal">
          ⚠️ Educational resource only — not a medical device. Always consult a licensed pharmacist or physician before changing medications.
        </p>
      </footer>

      {/* Floating Circle Theme Toggle Button - High Visibility */}
      <button
        onClick={() => setIsDark(!isDark)}
        className="fixed bottom-6 right-6 z-50 w-11 h-11 rounded-full bg-[var(--bg-card)] border border-[#E2DED8] dark:border-[#2C2C2E] text-[var(--text-primary)] shadow-md hover:shadow-lg transition-all hover:scale-105 cursor-pointer flex items-center justify-center focus:outline-none"
        aria-label="Toggle dark mode"
      >
        {isDark ? (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="5"></circle>
            <line x1="12" y1="1" x2="12" y2="3"></line>
            <line x1="12" y1="21" x2="12" y2="23"></line>
            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
            <line x1="1" y1="12" x2="3" y2="12"></line>
            <line x1="21" y1="12" x2="23" y2="12"></line>
            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
          </svg>
        ) : (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
          </svg>
        )}
      </button>
    </div>
  )
}
