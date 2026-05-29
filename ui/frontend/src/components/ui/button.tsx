import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded text-sm font-medium transition-colors disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default:
          "bg-ph-navy text-white hover:bg-ph-navy-dark active:bg-ph-navy-darker",
        destructive:
          "bg-ph-crimson text-white hover:bg-ph-crimson-dark active:bg-ph-crimson-darker",
        outline:
          "bg-white text-neutral-900 border border-neutral-300 hover:bg-neutral-50 active:bg-neutral-100",
        secondary:
          "bg-white text-neutral-900 border border-neutral-300 hover:bg-neutral-50 active:bg-neutral-100",
        ghost:
          "bg-transparent text-neutral-700 hover:bg-neutral-100 active:bg-neutral-200",
        link: "bg-transparent text-ph-navy hover:underline px-0 h-auto",
      },
      size: {
        default: "h-9 px-4 text-sm",
        sm: "h-8 px-3 text-xs",
        lg: "h-10 px-5 text-sm",
        icon: "h-9 w-9",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }
