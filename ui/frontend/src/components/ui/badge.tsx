import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center gap-1.5 px-2 py-0.5 rounded-sm text-xs font-medium border whitespace-nowrap transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        // Tone-based variants
        neutral: "bg-neutral-100 text-neutral-700 border-neutral-200",
        success: "bg-success-bg text-success border-success-border",
        warning: "bg-warning-bg text-warning border-warning-border",
        danger: "bg-danger-bg text-danger border-danger-border",
        info: "bg-info-bg text-info border-info-border",
        brand: "bg-neutral-100 text-ph-navy border-neutral-200",
        // Backward-compatibility aliases
        outline: "bg-neutral-100 text-neutral-700 border-neutral-200",
        secondary: "bg-neutral-100 text-neutral-700 border-neutral-200",
        destructive: "bg-danger-bg text-danger border-danger-border",
        default: "bg-neutral-100 text-ph-navy border-neutral-200",
      },
    },
    defaultVariants: {
      variant: "neutral",
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}

export { Badge, badgeVariants }
