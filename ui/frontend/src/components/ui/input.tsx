import * as React from "react"

import { cn } from "@/lib/utils"

const Input = React.forwardRef<HTMLInputElement, React.ComponentProps<"input">>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          "flex h-9 w-full rounded border border-neutral-300 bg-white px-3 py-2 text-sm text-neutral-900 transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-foreground placeholder:text-neutral-400 focus-visible:outline-none focus-visible:border-ph-navy focus-visible:ring-1 focus-visible:ring-ph-navy disabled:cursor-not-allowed disabled:bg-neutral-50 disabled:text-neutral-400 md:text-sm",
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)
Input.displayName = "Input"

export { Input }
