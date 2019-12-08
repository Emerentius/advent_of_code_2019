use itertools::Itertools;

enum Part {
    One = 1,
    Two = 2,
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

const NOUN_PTR: usize = 1;
const VERB_PTR: usize = 2;
const DAY_2_PART_2_REQUIRED_OUTPUT: i64 = 19690720;

fn day_2(part: crate::Part) {
    let input: &str = include_str!("day_2_input.txt");
    let mut memory = input
        .trim()
        .split(',')
        .map(str::parse::<i64>)
        .map(Result::<_, _>::unwrap)
        .collect::<Vec<_>>();

    match part {
        Part::One => {
            // >before running the program, replace position 1 with the value 12 and replace position 2 with the value 2
            memory[NOUN_PTR] = 12;
            memory[VERB_PTR] = 2;

            let result = run_intcode_program(memory);
            println!("{}", result);
        }
        Part::Two => {
            let (noun, verb) = (0..100)
                .flat_map(|noun| (0..100).map(move |verb| (noun, verb)))
                .find(|&(noun, verb)| {
                    memory[NOUN_PTR] = noun;
                    memory[VERB_PTR] = verb;
                    run_intcode_program(memory.clone()) == DAY_2_PART_2_REQUIRED_OUTPUT
                })
                .unwrap();
            println!("{}", noun * 100 + verb);
        }
    }
}

fn run_intcode_program(mut memory: Vec<i64>) -> i64 {
    // instruction pointer
    let mut instr_ptr = 0;
    loop {
        let opcode = memory[instr_ptr];
        match opcode {
            1 | 2 => {
                if let &[pos1, pos2, result_pos] = &memory[instr_ptr + 1..instr_ptr + 4] {
                    let op = if opcode == 1 {
                        std::ops::Add::add
                    } else {
                        std::ops::Mul::mul
                    };
                    memory[result_pos as usize] = op(memory[pos1 as usize], memory[pos2 as usize]);
                } else {
                    unreachable!();
                }
                instr_ptr += 4;
            }
            99 => break,
            _ => unreachable!(),
        }
    }
    memory[0]
}

// ===============================================================================================
//                                      Day 3
// ===============================================================================================
fn day_3(part: Part) {
    match part {
        Part::One => day_3_part_1(),
        Part::Two => day_3_part_2(),
    }
}

fn day_3_part_1() {
    use std::collections::{HashMap, HashSet};
    let input: &str = include_str!("day_3_input.txt");

    let mut wires_on_cell = HashMap::new();

    for (wire_nr, wire) in input.lines().enumerate() {
        let mut x: i32 = 0;
        let mut y: i32 = 0;
        for segment in wire.split(',') {
            let (direction, steps) = segment.split_at(1);
            let n_steps = steps.parse::<u64>().unwrap();
            let (dx, dy) = match direction {
                "R" => (1, 0),
                "L" => (-1, 0),
                "U" => (0, 1),
                "D" => (0, -1),
                _ => unreachable!(),
            };

            for _ in 0..n_steps {
                x += dx;
                y += dy;
                wires_on_cell
                    .entry((x, y))
                    .or_insert_with(HashSet::new)
                    .insert(wire_nr);
            }
        }
    }

    let distance = wires_on_cell.into_iter()
        .map(|((x, y), wires)| (x, y, wires.len()))
        .filter(|&(_, _, n_wires)| n_wires > 1)
        .map(|(x, y, _)| x.abs() + y.abs())
        .min()
        .unwrap();
    println!("{}", distance);
}

fn day_3_part_2() {
    use std::collections::HashMap;
    let input: &str = include_str!("day_3_input.txt");

    let mut wires_on_cell = HashMap::new();

    for (wire_nr, wire) in input.lines().enumerate() {
        let mut x: i32 = 0;
        let mut y: i32 = 0;

        let mut distance_traveled: u32 = 0;
        for segment in wire.split(',') {
            let (direction, steps) = segment.split_at(1);
            let n_steps = steps.parse::<u64>().unwrap();
            let (dx, dy) = match direction {
                "R" => (1, 0),
                "L" => (-1, 0),
                "U" => (0, 1),
                "D" => (0, -1),
                _ => unreachable!(),
            };

            for _ in 0..n_steps {
                x += dx;
                y += dy;
                distance_traveled += 1;

                // Store distance traveled, but only the first time we
                // visit this cell to keep the minimum distance.
                wires_on_cell
                    .entry((x, y))
                    .or_insert_with(HashMap::new)
                    .entry(wire_nr)
                    .or_insert(distance_traveled);
            }
        }
    }

    let distance: u32 = wires_on_cell.into_iter()
        .map(|(_, wire_distances)| wire_distances)
        .filter(|wire_distances| wire_distances.len() > 1)
        .map(|wire_distances| wire_distances.values().sum())
        .min()
        .unwrap();
    println!("day 3 part 2, fewest combined steps the wires must take to reach an intersection:\n{}", distance);
}

// ===============================================================================================
//                                      Day 4
// ===============================================================================================

const DAY_4_PASSWORD_RANGE: std::ops::RangeInclusive<u32> = 273025..=767253;

// from least to most significant
fn digits(mut num: u32) -> impl Iterator<Item=u32> {
    std::iter::from_fn(move || {
        if num != 0 {
            let digit = num % 10;
            num /= 10;
            Some(digit)
        } else {
            None
        }
    })
}

fn digits_increase_monotonically(&num: &u32) -> bool {
    digits(num)
        .tuple_windows()
        .all(|(digit, more_significant_digit)| more_significant_digit <= digit)
}

fn two_adjacent_digits_are_equal(&num: &u32) -> bool {
    digits(num)
        .tuple_windows()
        .any(|(digit1, digit2)| digit1 == digit2)
}

fn exactly_two_adjacent_digits_are_equal(&num: &u32) -> bool {
    let grouped_digits = digits(num).group_by(|&digit| digit);
    grouped_digits
        .into_iter()
        .any(|(_, group)| group.count() == 2)
}

fn day_4(part: Part) {
    // less than a million numbers
    // => easy bruteforce
    let adjacent_number_filter = match part {
        Part::One => two_adjacent_digits_are_equal,
        Part::Two => exactly_two_adjacent_digits_are_equal,
    };

    let count = DAY_4_PASSWORD_RANGE
        .filter(digits_increase_monotonically)
        .filter(adjacent_number_filter)
        .count();

    println!("day 4 part {}: nr of passwords in the range fitting the criteria\n{}", part as i32, count)
}

fn main() {
    // keep old code in here to avoid unused function warnings
    if false {
        day_1(Part::One);
        day_1(Part::Two);
        day_2(Part::One);
        day_2(Part::Two);
        day_3(Part::One);
        day_3(Part::Two);
        day_4(Part::One);
    }
    day_4(Part::Two);
}
