import React from 'react';

export default function Button({ children, variant = 'primary', className = '', as = 'button', danger = false, ...props }) {
  const base = 'btn';
  const variantClass = variant === 'primary' ? 'btn-primary' : variant === 'outline' ? 'btn-outline' : 'btn-ghost';
  const dangerClass = danger ? ' text-red-600 border-red-100' : '';
  const full = `${base} ${variantClass} ${dangerClass} ${className}`.trim();

  if (as && as !== 'button') {
    const Tag = as;
    return (
      <Tag className={full} {...props}>
        {children}
      </Tag>
    );
  }

  return (
    <button className={full} {...props}>
      {children}
    </button>
  );
}
