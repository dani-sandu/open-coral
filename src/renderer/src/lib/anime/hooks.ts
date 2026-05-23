import { useRef, useEffect, useState, type RefObject } from 'react'
import { animate, type JSAnimation } from 'animejs'
import {
  ANIMATION_ENABLED,
  SPRING_TOKEN, SPRING_STATUS, SPRING_PEER,
  DURATION_TOKEN, DURATION_PEER,
} from './config'

type AnimateParams = Parameters<typeof animate>[1]

// General-purpose: returns [ref, animateFn]. animateFn() no-ops when ANIMATION_ENABLED is false.
export function useAnimeRef<T extends Element = HTMLElement>(): [RefObject<T | null>, (params: AnimateParams) => JSAnimation | undefined] {
  const ref = useRef<T>(null)
  const instanceRef = useRef<JSAnimation | null>(null)

  const animateFn = (params: AnimateParams): JSAnimation | undefined => {
    if (!ANIMATION_ENABLED || !ref.current) return
    instanceRef.current = animate(ref.current, params)
    return instanceRef.current
  }

  useEffect(() => () => { instanceRef.current?.cancel() }, [])
  return [ref, animateFn]
}

// Status row: slides element in (translateY 20→0, opacity 0→1) when active flips true.
export function useStatusRow(active: boolean): RefObject<HTMLElement | null> {
  const ref = useRef<HTMLElement>(null)
  const instanceRef = useRef<JSAnimation | null>(null)

  useEffect(() => {
    if (!active || !ref.current || !ANIMATION_ENABLED) return
    instanceRef.current = animate(ref.current, {
      opacity: [0, 1],
      translateY: [20, 0],
      ease: SPRING_STATUS,
    })
    return () => { instanceRef.current?.cancel() }
  }, [active])

  return ref
}

// Fade slide: fires once on mount. Attach to each newly rendered message element.
export function useFadeSlide(): RefObject<HTMLElement | null> {
  const ref = useRef<HTMLElement>(null)
  const instanceRef = useRef<JSAnimation | null>(null)

  useEffect(() => {
    if (!ref.current || !ANIMATION_ENABLED) return
    instanceRef.current = animate(ref.current, {
      opacity: [0, 1],
      translateY: [4, 0],
      ease: SPRING_TOKEN,
      duration: DURATION_TOKEN,
    })
    return () => { instanceRef.current?.cancel() }
  }, [])

  return ref
}

// Pulsing dot: loops opacity while active, stops cleanly on deactivate.
export function usePulsingDot(active: boolean): RefObject<HTMLElement | null> {
  const ref = useRef<HTMLElement>(null)
  const instanceRef = useRef<JSAnimation | null>(null)

  useEffect(() => {
    if (!ref.current || !ANIMATION_ENABLED) return
    if (active) {
      instanceRef.current = animate(ref.current, {
        opacity: [1, 0.3],
        loop: true,
        alternate: true,
        ease: 'inOutSine',
        duration: 900,
      })
    } else {
      instanceRef.current?.cancel()
      instanceRef.current = animate(ref.current, { opacity: 1, duration: 200 })
    }
    return () => { instanceRef.current?.cancel() }
  }, [active])

  return ref
}

const SCRAMBLE_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

// Scramble text: chars cycle randomly, then resolve left-to-right. Loops continuously.
export function useScrambleText(text: string): string {
  const n = text.length

  const scramble = (p: number) =>
    text.split('').map((char, i) => {
      if (!/[a-zA-Z0-9]/.test(char)) return char
      const resolveAt = 0.5 + (i / n) * 0.5
      if (p >= resolveAt) return char
      return SCRAMBLE_CHARS[Math.floor(Math.random() * SCRAMBLE_CHARS.length)]
    }).join('')

  const [displayed, setDisplayed] = useState(() => ANIMATION_ENABLED ? scramble(0) : text)
  const instanceRef = useRef<JSAnimation | null>(null)

  useEffect(() => {
    if (!ANIMATION_ENABLED) return
    let timeoutId: ReturnType<typeof setTimeout>

    const run = () => {
      const progress = { value: 0 }
      instanceRef.current = animate(progress, {
        value: 1,
        duration: 1400,
        ease: 'outCubic',
        onUpdate: () => setDisplayed(scramble(progress.value)),
        onComplete: () => {
          setDisplayed(text)
          timeoutId = setTimeout(run, 2500)
        },
      })
    }

    run()
    return () => { instanceRef.current?.cancel(); clearTimeout(timeoutId) }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return displayed
}

// SVG peer node: scale 0→1 + opacity 0→1 on mount when isNew is true.
export function usePeerEntrance(isNew: boolean): RefObject<SVGGElement | null> {
  const ref = useRef<SVGGElement>(null)
  const instanceRef = useRef<JSAnimation | null>(null)

  useEffect(() => {
    if (!isNew || !ref.current || !ANIMATION_ENABLED) return
    instanceRef.current = animate(ref.current, {
      opacity: [0, 1],
      scale: [0, 1],
      ease: SPRING_PEER,
      duration: DURATION_PEER,
    })
    return () => { instanceRef.current?.cancel() }
  }, [])

  return ref
}
