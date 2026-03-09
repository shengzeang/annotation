import React from 'react';

export default function Button({ children, variant = 'primary', className = '', as = 'button', danger = false, ...props }) {
  const variantClass =
    danger ? 'btn-danger' :
    variant === 'primary' ? 'btn-primary' :
    variant === 'secondary' ? 'btn-secondary' :
    variant === 'outline' ? 'btn-secondary' :
    'btn-ghost';

  const full = `btn ${variantClass} ${className}`.trim();

  if (as && as !== 'button') {
    const Tag = as;
    return <Tag className={full} {...props}>{children}</Tag>;
  }

  return <button className={full} {...props}>{children}</button>;
}
