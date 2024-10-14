use csv::Writer;
use matrix::Matrix;
use num_traits::Float;
use std::{fmt::Display, io::Write};

fn table_row<T: Float + Display>(x: &Matrix<T>, t: T) -> Vec<String> {
    let mut line = vec![format!("{:.3}", t)];
    line.append(
        &mut x
            .to_slice()
            .iter()
            .map(|elm| format!("{:.4}", elm))
            .collect(),
    );
    line
}

fn write_result<T: Write>(table: &Vec<Vec<String>>, result_file: &mut Writer<T>) {
    for line in table {
        let _ = result_file.write_record(line);
    }
}

#[cfg(feature = "rayon")]
pub trait EulerSolver<T, S>
where
    T: Float + Display + Send + Sync,
    S: Write,
{
    fn solve(&self, x0: &Matrix<T>, dt: T, t_end: T, result_file: &mut Writer<S>) {
        let mut t = T::zero();
        let mut x = x0.clone();
        let mut table = vec![table_row(&x, t)];

        while t <= t_end {
            let dx = Self::dot_x(self, &x, t) * dt;

            x = self.post_process(&(&x + dx));
            t = t + dt;

            table.push(table_row(&x, t));
        }

        write_result(&table, result_file);
    }

    fn dot_x(&self, x: &Matrix<T>, t: T) -> Matrix<T>;

    fn post_process(&self, x: &Matrix<T>) -> Matrix<T>;
}

#[cfg(not(feature = "rayon"))]
pub trait EulerSolver<T, S>
where
    T: Float + Display,
    S: Write,
{
    fn solve(&self, x0: &Matrix<T>, dt: T, t_end: T, result_file: &mut Writer<S>) {
        let mut t = T::zero();
        let mut x = x0.clone();
        let mut table = vec![table_row(&x, t)];

        while t <= t_end {
            let dx = self.dot_x(&x, t) * dt;

            x = self.post_process(&(&x + dx));
            t = t + dt;

            table.push(table_row(&x, t));
        }

        write_result(&table, result_file);
    }

    fn dot_x(&self, x: &Matrix<T>, t: T) -> Matrix<T>;

    fn post_process(&self, x: &Matrix<T>) -> Matrix<T>;
}

#[cfg(feature = "rayon")]
pub trait RungeKuttaSolver<T, S>
where
    T: Float + Display + Send + Sync,
    S: Write,
{
    fn solve(&self, x0: &Matrix<T>, dt: T, t_end: T, result_file: &mut Writer<S>) {
        let mut t = T::zero();
        let mut x = x0.clone();
        let mut table = vec![table_row(&x, t)];

        let two = T::one() + T::one();
        let half = T::one() / two;
        let six = two + two + two;

        while t <= t_end {
            let k1 = Self::dot_x(self, &x, t) * dt;
            let k2 = Self::dot_x(self, &(&x + &k1 * half), t + dt * half) * dt;
            let k3 = Self::dot_x(self, &(&x + &k2 * half), t + dt * half) * dt;
            let k4 = Self::dot_x(self, &(&x + &k3), t + dt) * dt;

            let dx = (k1 + k2 * two + k3 * two + k4) / six;

            x = self.post_process(&(&x + dx));
            t = t + dt;

            table.push(table_row(&x, t));
        }

        write_result(&table, result_file);
    }

    fn dot_x(&self, x: &Matrix<T>, t: T) -> Matrix<T>;

    fn post_process(&self, x: &Matrix<T>) -> Matrix<T>;
}

#[cfg(not(feature = "rayon"))]
pub trait DelayedEulerSolver<T, S>
where
    T: Float + Display,
    S: Write,
{
    fn solve(&mut self, x0: &Matrix<T>, dt: T, t_end: T, result_file: &mut Writer<S>) {
        let mut t = T::zero();
        let mut k = 0;
        let mut x = x0.clone();
        let mut table = vec![table_row(&x, t)];

        while t <= t_end {
            let dx = self.dot_x(&x, t, k) * dt;
            self.push_buffer(&x, k);

            x = self.post_process(&(&x + dx));
            t = t + dt;
            k += 1;

            table.push(table_row(&x, t));
        }

        write_result(&table, result_file);
    }

    fn dot_x(&self, x: &Matrix<T>, t: T, k: usize) -> Matrix<T>;

    fn post_process(&self, x: &Matrix<T>) -> Matrix<T>;

    fn push_buffer(&mut self, x: &Matrix<T>, k: usize);
}

#[cfg(not(feature = "rayon"))]
pub trait RungeKuttaSolver<T, S>
where
    T: Float + Display,
    S: Write,
{
    fn solve(&self, x0: &Matrix<T>, dt: T, t_end: T, result_file: &mut Writer<S>) {
        let mut t = T::zero();
        let mut x = x0.clone();
        let mut table = vec![table_row(&x, t)];

        let two = T::one() + T::one();
        let half = T::one() / two;
        let six = two + two + two;

        while t <= t_end {
            let k1 = self.dot_x(&x, t) * dt;
            let k2 = self.dot_x(&(&x + &k1 * half), t + dt * half) * dt;
            let k3 = self.dot_x(&(&x + &k2 * half), t + dt * half) * dt;
            let k4 = self.dot_x(&(&x + &k3), t + dt) * dt;

            let dx = (k1 + k2 * two + k3 * two + k4) / six;

            x = self.post_process(&(&x + dx));
            t = t + dt;

            table.push(table_row(&x, t));
        }

        write_result(&table, result_file);
    }

    fn dot_x(&self, x: &Matrix<T>, t: T) -> Matrix<T>;

    fn post_process(&self, x: &Matrix<T>) -> Matrix<T>;
}