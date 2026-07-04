import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { PatientAnalyzerPage } from './pages/PatientAnalyzerPage'
import { PairwiseLookupPage } from './pages/PairwiseLookupPage'
import { AlternativesPage } from './pages/AlternativesPage'
import { MolecularExplorerPage } from './pages/MolecularExplorerPage'
import { DirectoryPage } from './pages/DirectoryPage'
import { NeighborhoodPage } from './pages/NeighborhoodPage'
import { AboutPage } from './pages/AboutPage'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      staleTime: 5 * 60 * 1000, // 5 min
    },
  },
})

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<PatientAnalyzerPage />} />
            <Route path="pairwise" element={<PairwiseLookupPage />} />
            <Route path="alternatives" element={<AlternativesPage />} />
            <Route path="molecular" element={<MolecularExplorerPage />} />
            <Route path="directory" element={<DirectoryPage />} />
            <Route path="neighborhood" element={<NeighborhoodPage />} />
            <Route path="about" element={<AboutPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}

export default App
