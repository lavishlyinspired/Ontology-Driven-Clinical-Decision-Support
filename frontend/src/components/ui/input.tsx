"use client";

import * as React from "react";

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className = "", type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={`flex h-10 w-full rounded-md border border-[#1e293b] bg-[#0f172a] px-3 py-2 text-sm text-[#f8fafc] placeholder:text-[#64748b] focus:outline-none focus:ring-2 focus:ring-[#63e6be] focus:ring-offset-2 focus:ring-offset-[#030711] disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
        ref={ref}
        {...props}
      />
    );
  }
);
Input.displayName = "Input";

export { Input };
