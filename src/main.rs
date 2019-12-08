enum Part {
    One,
    Two,
}

// ===============================================================================================
//                                      Day 1
// ===============================================================================================
fn naive_fuel_for_module(mass: i64) -> i64 {
    mass / 3 - 2
}

fn fuel_for_module(mass: i64) -> i64 {
    std::iter::successors(Some(mass), |&mass| Some(naive_fuel_for_module(mass)))
        .skip(1) // module mass itself is not fuel
        .take_while(|&fuel| fuel > 0)
        .sum()
}

fn day_1(part: crate::Part) {
    static INPUT: &str = include_str!("day_1_input.txt");
    let fuel_for_module = match part {
        Part::One => naive_fuel_for_module,
        Part::Two => fuel_for_module,
    };

    let total_fuel = INPUT
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| line.parse::<i64>().unwrap())
        .map(fuel_for_module)
        .sum::<i64>();

    println!("{}", total_fuel);
}

#[test]
fn fuel_required() {
    for &(mass, fuel) in [(12, 2), (14, 2), (1969, 654), (100756, 33583)].iter() {
        assert_eq!(naive_fuel_for_module(mass), fuel);
    }
}

// ===============================================================================================
//                                      Day 2
// ===============================================================================================

fn day_2(_part: crate::Part) {
    let input: &str = include_str!("day_2_input.txt");
    let mut memory = input
        .trim()
        .split(',')
        .map(str::parse::<i64>)
        .map(Result::<_, _>::unwrap)
        .collect::<Vec<_>>();
    let mut pos = 0;

    // >before running the program, replace position 1 with the value 12 and replace position 2 with the value 2
    memory[1] = 12;
    memory[2] = 2;

    loop {
        let opcode = memory[pos];
        match opcode {
            1 | 2 => {
                if let &[pos1, pos2, result_pos] = &memory[pos + 1..pos + 4] {
                    let op = if opcode == 1 {
                        std::ops::Add::add
                    } else {
                        std::ops::Mul::mul
                    };
                    memory[result_pos as usize] = op(memory[pos1 as usize], memory[pos2 as usize]);
                } else {
                    unreachable!();
                }
                pos += 4;
            }
            99 => break,
            _ => unreachable!(),
        }
    }
    println!("{}", memory[0]);
}

fn main() {
    if false {
        day_1(Part::One);
        day_1(Part::Two);
    }

    day_2(Part::One);
}
