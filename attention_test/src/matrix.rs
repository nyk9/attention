use std::{f64, fmt};

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub columns: usize,
    data: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn new(rows: usize, columns: usize) -> Self {
        Matrix {
            rows,
            columns,
            data: vec![vec![0.0; columns]; rows],
        }
    }

    #[allow(dead_code)]
    pub fn zeros(rows: usize, columns: usize) -> Self {
        Self::new(rows, columns)
    }

    pub fn from_vec(data: Vec<Vec<f64>>) -> Self {
        if !data.is_empty() {
            for row in &data {
                if row.len() != data[0].len() {
                    panic!("All rows must have the same length");
                }
            }
        }
        Matrix {
            rows: data.len(),
            columns: data[0].len(),
            data,
        }
    }

    pub fn multiply(&self, other: &Matrix) -> Matrix {
        if self.columns != other.rows {
            panic!(
                "次元が一致しません: ({}, {}) × ({}, {})",
                self.rows, self.columns, other.rows, other.columns
            );
        }
        let mut result = Matrix::new(self.rows, other.columns);

        for i in 0..self.rows {
            for j in 0..other.columns {
                let mut sum = 0.0;
                for k in 0..self.columns {
                    sum += self.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        result
    }

    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.columns, self.rows);

        for i in 0..self.rows {
            for j in 0..self.columns {
                result.data[j][i] = self.data[i][j];
            }
        }

        result
    }

    pub fn get_row(&self, row_index: usize) -> Vec<f64> {
        self.data[row_index].clone()
    }

    pub fn scale(&self, scalar: f64) -> Matrix {
        let mut result_data = Vec::new();
        for row in &self.data {
            let scaled_row: Vec<f64> = row.iter().map(|&val| val / scalar).collect();
            result_data.push(scaled_row);
        }
        Matrix::from_vec(result_data)
    }

    pub fn set(&mut self, row: usize, column: usize, value: f64) {
        self.data[row][column] = value;
    }

    pub fn get(&self, row: usize, column: usize) -> f64 {
        self.data[row][column]
    }

    // 要素ごとの積 （Hadamard積、A ⊙ B）
    #[allow(dead_code)]
    pub fn hadamard(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.columns != other.columns {
            panic!(
                "次元が一致しません: ({}, {}) ⊙ ({}, {})",
                self.rows, self.columns, other.rows, other.columns
            );
        }

        let mut result = Matrix::new(self.rows, self.columns);
        for i in 0..self.rows {
            for j in 0..self.columns {
                result.data[i][j] = self.data[i][j] * other.data[i][j];
            }
        }
        result
    }

    // 行列の減算　（A - B）
    #[allow(dead_code)]
    pub fn subtract(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.columns != other.columns {
            panic!(
                "次元が一致しません: ({}, {}) ⊙ ({}, {})",
                self.rows, self.columns, other.rows, other.columns
            );
        }

        let mut result = Matrix::new(self.rows, self.columns);
        for i in 0..self.rows {
            for j in 0..self.columns {
                result.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }
        result
    }

    // 行列の加算 （A + B）
    pub fn add(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.columns != other.columns {
            panic!(
                "次元が一致しません: ({}, {}) ⊙ ({}, {})",
                self.rows, self.columns, other.rows, other.columns
            );
        }

        let mut result = Matrix::new(self.rows, self.columns);
        for i in 0..self.rows {
            for j in 0..self.columns {
                result.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        result
    }

    // スカラー倍 （c × A）
    #[allow(dead_code)]
    pub fn scalar_multiply(&self, scalar: f64) -> Matrix {
        let mut result = Matrix::new(self.rows, self.columns);

        for i in 0..self.rows {
            for j in 0..self.columns {
                result.data[i][j] = self.data[i][j] * scalar;
            }
        }
        result
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Matrix ({}×{}):", self.rows, self.columns)?;
        for row in &self.data {
            writeln!(f, "{:?}", row)?;
        }
        Ok(())
    }
}

pub fn soft_max(input: &Vec<f64>) -> Vec<f64> {
    let max_val = input.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let mut exp_values = Vec::new();
    let mut sum = 0.0;

    for &val in input {
        let exp_val = (val - max_val).exp();
        exp_values.push(exp_val);
        sum += exp_val;
    }

    let result: Vec<f64> = exp_values.iter().map(|&exp_val| exp_val / sum).collect();

    result
}

// ReLU活性化関数: max(0, x)
// 各要素に対して適用
pub fn relu(input: &Matrix) -> Matrix {
    let mut result = Matrix::new(input.rows, input.columns);

    for i in 0..input.rows {
        for j in 0..input.columns {
            result.data[i][j] = input.data[i][j].max(0.0);
        }
    }

    result
}

// ReLU勾配マスク: x > 0 なら 1.0、そうでなければ 0.0
// 逆伝播で使用（要素ごとに勾配を通過させるか決定）
pub fn relu_gradient(input: &Matrix) -> Matrix {
    let mut result = Matrix::new(input.rows, input.columns);

    for i in 0..input.rows {
        for j in 0..input.columns {
            result.data[i][j] = if input.data[i][j] > 0.0 { 1.0 } else { 0.0 };
        }
    }

    result
}
