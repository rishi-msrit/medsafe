import { useEffect, useRef, useState } from 'react'
import type { SafetyReport } from '../api'

interface RiskScoreRingProps {
  report: SafetyReport
  size?: number
}

const TIER_GRADIENT_ID = 'risk-ring-gradient'

export function RiskScoreRing({ report, size = 180 }: RiskScoreRingProps) {
  const score = report.overall_risk_score
  const tier = report.risk_tier
  const tierColor = report.risk_tier_color

  const radius = size * 0.38
  const cx = size / 2
  const cy = size / 2
  const circumference = 2 * Math.PI * radius
  const strokeWidth = size * 0.06

  // Animate from 0 to score
  const [displayScore, setDisplayScore] = useState(0)
  const [dashOffset, setDashOffset] = useState(circumference)
  const animRef = useRef<number | null>(null)

  useEffect(() => {
    const duration = 1200
    const start = performance.now()
    const animate = (now: number) => {
      const t = Math.min((now - start) / duration, 1)
      const ease = 1 - Math.pow(1 - t, 3) // Cubic ease-out
      const current = score * ease
      setDisplayScore(Math.round(current))
      setDashOffset(circumference - (current / 100) * circumference)
      if (t < 1) {
        animRef.current = requestAnimationFrame(animate)
      }
    }
    animRef.current = requestAnimationFrame(animate)
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current) }
  }, [score, circumference])

  const tierConfig = {
    safe:     { label: 'Generally Safe',   textColor: 'text-green-600' },
    review:   { label: 'Review Recommended',textColor: 'text-amber-600' },
    high:     { label: 'High Risk',         textColor: 'text-orange-600' },
    critical: { label: 'Critical',          textColor: 'text-red-600' },
  }

  const config = tierConfig[tier] || tierConfig.review

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="transform -rotate-90">
          <defs>
            <linearGradient id={TIER_GRADIENT_ID} x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor={tierColor} stopOpacity="0.8" />
              <stop offset="100%" stopColor={tierColor} stopOpacity="1.0" />
            </linearGradient>
          </defs>
          {/* Track */}
          <circle
            cx={cx} cy={cy} r={radius}
            fill="none"
            stroke="rgba(15, 23, 42, 0.08)"
            strokeWidth={strokeWidth}
          />
          {/* Progress */}
          <circle
            cx={cx} cy={cy} r={radius}
            fill="none"
            stroke={`url(#${TIER_GRADIENT_ID})`}
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
            strokeLinecap="round"
            className="score-ring"
          />
        </svg>

        {/* Center Text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-4xl font-black ${config.textColor}`}>
            {displayScore}
          </span>
          <span className="text-xs text-slate-400 font-semibold mt-0.5">/ 100</span>
        </div>
      </div>

      <div className="text-center">
        <div className={`text-sm font-bold ${config.textColor}`}>
          {report.risk_tier_label}
        </div>
        <div className="text-xs text-slate-500 font-medium mt-0.5">
          {report.num_pairs_checked} pairs analyzed
        </div>
      </div>
    </div>
  )
}
