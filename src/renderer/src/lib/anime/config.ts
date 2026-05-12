import { createSpring } from 'animejs'

const reducedMotion =
  typeof window !== 'undefined' &&
  window.matchMedia('(prefers-reduced-motion: reduce)').matches

export const ANIMATION_ENABLED = !reducedMotion

// Spring easings — anime.js v4 createSpring
export const SPRING_TOKEN  = createSpring({ stiffness: 120, damping: 18 }) // fast, minimal bounce
export const SPRING_STATUS = createSpring({ stiffness: 100, damping: 14 }) // medium, soft entry
export const SPRING_PEER   = createSpring({ stiffness:  80, damping: 10 }) // slow, visible overshoot

// Durations in ms (used for non-spring animations)
export const DURATION_TOKEN    = 240
export const DURATION_STATUS   = 320
export const DURATION_PROGRESS = 400
export const DURATION_PEER     = 400
