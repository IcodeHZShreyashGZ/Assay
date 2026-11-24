-- Assay Production Demo Database Seed (PostgreSQL)
-- Schema modeling a SaaS / Subscription Analytics domain aligned with demo/glossary.yaml

CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150) UNIQUE NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'active', -- 'active', 'churned', 'trial'
    total_orders INT DEFAULT 0,
    last_login_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS subscriptions (
    subscription_id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(user_id) ON DELETE CASCADE,
    plan_name VARCHAR(50) NOT NULL,
    monthly_price NUMERIC(10, 2) NOT NULL,
    starting_mrr NUMERIC(10, 2) NOT NULL,
    ending_mrr NUMERIC(10, 2) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS orders (
    order_id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(user_id) ON DELETE CASCADE,
    amount NUMERIC(10, 2) NOT NULL,
    quantity INT DEFAULT 1,
    rating NUMERIC(2, 1),
    status VARCHAR(20) NOT NULL DEFAULT 'completed',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id SERIAL PRIMARY KEY,
    order_id INT REFERENCES orders(order_id) ON DELETE CASCADE,
    user_id INT REFERENCES users(user_id) ON DELETE CASCADE,
    amount NUMERIC(10, 2) NOT NULL,
    payment_method VARCHAR(50) DEFAULT 'credit_card',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS marketing_spend (
    spend_id SERIAL PRIMARY KEY,
    channel VARCHAR(50) NOT NULL,
    spend_amount NUMERIC(10, 2) NOT NULL,
    new_customer_id INT REFERENCES users(user_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Seed Data
INSERT INTO users (name, email, status, total_orders, last_login_at, created_at) VALUES
('Alice Walker', 'alice@example.com', 'active', 5, CURRENT_TIMESTAMP - INTERVAL '2 days', CURRENT_TIMESTAMP - INTERVAL '60 days'),
('Bob Smith', 'bob@example.com', 'active', 2, CURRENT_TIMESTAMP - INTERVAL '10 days', CURRENT_TIMESTAMP - INTERVAL '45 days'),
('Charlie Brown', 'charlie@example.com', 'churned', 1, CURRENT_TIMESTAMP - INTERVAL '40 days', CURRENT_TIMESTAMP - INTERVAL '90 days'),
('Diana Prince', 'diana@example.com', 'active', 12, CURRENT_TIMESTAMP - INTERVAL '1 hour', CURRENT_TIMESTAMP - INTERVAL '120 days'),
('Evan Wright', 'evan@example.com', 'churned', 0, CURRENT_TIMESTAMP - INTERVAL '50 days', CURRENT_TIMESTAMP - INTERVAL '70 days'),
('Fiona Gallagher', 'fiona@example.com', 'active', 4, CURRENT_TIMESTAMP - INTERVAL '5 days', CURRENT_TIMESTAMP - INTERVAL '30 days');

INSERT INTO subscriptions (user_id, plan_name, monthly_price, starting_mrr, ending_mrr, status) VALUES
(1, 'Professional', 99.00, 99.00, 99.00, 'active'),
(2, 'Starter', 29.00, 29.00, 29.00, 'active'),
(3, 'Starter', 29.00, 29.00, 0.00, 'cancelled'),
(4, 'Enterprise', 499.00, 499.00, 599.00, 'active'),
(6, 'Professional', 99.00, 99.00, 99.00, 'active');

INSERT INTO orders (user_id, amount, quantity, rating, status) VALUES
(1, 99.00, 1, 4.8, 'completed'),
(2, 29.00, 1, 4.2, 'completed'),
(3, 29.00, 1, 3.5, 'completed'),
(4, 499.00, 1, 5.0, 'completed'),
(4, 150.00, 3, 4.9, 'completed'),
(6, 99.00, 1, 4.7, 'completed');

INSERT INTO transactions (order_id, user_id, amount, payment_method) VALUES
(1, 1, 99.00, 'credit_card'),
(2, 2, 29.00, 'paypal'),
(3, 3, 29.00, 'credit_card'),
(4, 4, 499.00, 'wire_transfer'),
(5, 4, 150.00, 'credit_card'),
(6, 6, 99.00, 'credit_card');

INSERT INTO marketing_spend (channel, spend_amount, new_customer_id) VALUES
('google_ads', 150.00, 1),
('facebook_ads', 75.00, 2),
('linkedin_ads', 200.00, 4),
('organic_seo', 20.00, 6);
