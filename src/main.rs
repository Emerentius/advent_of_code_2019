
#![feature(test)]
extern crate test;

use itertools::Itertools;
use std::collections::VecDeque;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::BinaryHeap;
use std::convert::{TryInto, TryFrom};

macro_rules! puzzle_input {
    ($day: expr) => {
        include_str!( concat!("day_", stringify!($day), "_input.txt" ) )
    };
}

#[derive(Copy, Clone, PartialEq)]
enum Part {
    One = 1,
    Two = 2,
}

#[derive(PartialEq)]
enum ProgramState {
    RequiresInput,
    Finished,
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
    static INPUT: &str = puzzle_input!(1);
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
    let input = puzzle_input!(2);
    let mut memory = Program::parse_memory(input);

    match part {
        Part::One => {
            // >before running the program, replace position 1 with the value 12 and replace position 2 with the value 2
            memory[NOUN_PTR] = 12;
            memory[VERB_PTR] = 2;

            let result = run_intcode_program(memory);
            println!("day 2 part 1: {}", result);
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
            println!("day 2 part 2: {}", noun * 100 + verb);
        }
    }
}

/// IntCode program
#[derive(Clone)]
struct Program {
    memory: Vec<i64>,
    instr_ptr: usize,
    // parameters in relative position mode access memory
    // at position relative_base + value
    relative_base: usize,
    input: VecDeque<i64>,
    output: Vec<i64>,

    // trace of (instr_ptr, instruction)
    // `instruction` includes the parameter modes
    // for debugging only
    instructions_trace: Vec<(usize, i64)>,
}

const POSITION_MODE: i64 = 0;
const IMMEDIATE_MODE: i64 = 1;
const RELATIVE_POSITION_MODE: i64 = 2;

// The computer's available memory should be much larger than the initial program.
// "much larger". Thank you, for this precise statement.
const INITIAL_MEMORY_SIZE: usize = 1_000_000;

impl Program {
    fn new(serialized_memory: &str) -> Self {
        Self::from_memory(Self::parse_memory(serialized_memory))
    }

    fn from_memory(mut memory: Vec<i64>) -> Self {
        // at least double the size is "much larger", right?
        assert!(2 * memory.len() < INITIAL_MEMORY_SIZE);
        memory.resize(INITIAL_MEMORY_SIZE, 0);
        Self {
            memory,
            instr_ptr: 0,
            relative_base: 0,
            input: VecDeque::new(),
            output: vec![],
            instructions_trace: vec![],
        }
    }

    fn parse_memory(serialized_memory: &str) -> Vec<i64> {
        serialized_memory
            .trim()
            .split(',')
            .map(str::parse::<i64>)
            .map(Result::<_, _>::unwrap)
            .collect()
    }

    fn execute(serialized_memory: &str, input: impl Into<VecDeque<i64>>) -> Vec<i64> {
        let mut program = Self::new(serialized_memory);
        program.input = input.into();
        program.run();
        program.output
    }

    #[allow(unused)]
    fn debug_execute(serialized_memory: &str, input: impl Into<VecDeque<i64>>) -> Self {
        let mut program = Self::new(serialized_memory);
        program.input = input.into();
        program.run();
        program
    }

    fn run(&mut self) -> ProgramState {
        loop {
            let instruction = self.memory[self.instr_ptr];
            let opcode = instruction % 100;
            self.instructions_trace.push((self.instr_ptr, instruction));

            let _old_instr_ptr = self.instr_ptr;
            match opcode {
                // 1 add
                // 2 multiply
                1 | 2 => {
                    let op = if opcode == 1 {
                        std::ops::Add::add
                    } else {
                        std::ops::Mul::mul
                    };
                    let val1 = self.param_val(0);
                    let val2 = self.param_val(1);
                    let result_pos = self.param_address(2);

                    self.memory[result_pos] = op(val1, val2);
                    self.instr_ptr += 4;
                }
                // input
                3 => {
                    let return_pos = self.param_address(0);
                    match self.input.pop_front() {
                        Some(value) => self.memory[return_pos] = value,
                        None => return ProgramState::RequiresInput,
                    };
                    self.instr_ptr += 2;
                }
                // output
                4 => {
                    self.output.push(self.param_val(0));
                    self.instr_ptr += 2;
                }
                // jump-if
                5 | 6 => {
                    // 5 jump if true, 6 jump if false
                    // xor-ing with opcode == 6 conditionally inverts the boolean
                    // thereby handling both
                    match (self.param_val(0) != 0) ^ (opcode == 6) {
                        true => self.instr_ptr = self.param_val(1) as usize,
                        false => self.instr_ptr += 3,
                    }
                }
                // comparisons
                7 | 8 => {
                    let comparator = match opcode {
                        7 => PartialOrd::lt,
                        8 => PartialOrd::eq,
                        _ => unreachable!(),
                    };
                    let return_pos = self.param_address(2);
                    self.memory[return_pos] = comparator(&self.param_val(0), &self.param_val(1)) as _;
                    self.instr_ptr += 4;
                }
                9 => {
                    let new_pos = self.relative_base as i64 + self.param_val(0);
                    self.relative_base = new_pos.try_into().unwrap();
                    self.instr_ptr += 2;
                }
                99 => break,
                _ => unreachable!(),
            }

            assert_ne!(_old_instr_ptr, self.instr_ptr);
        }
        ProgramState::Finished
    }

    fn param_val(&self, param_nr: u32) -> i64 {
        let address = self.param_address(param_nr);
        self.memory[address]
    }

    fn param_address(&self, param_nr: u32) -> usize {
        let param_offset = param_nr as usize + 1;
        let param_idx = self._param_idx(param_offset);
        let address = match self._param_mode(param_nr) {
            POSITION_MODE => self.memory[param_idx] as usize,
            IMMEDIATE_MODE => param_idx,
            RELATIVE_POSITION_MODE => {
                let pos = self.relative_base as i64 + self.memory[param_idx];
                pos.try_into().unwrap()
            },
            _ => unreachable!(),
        };
        address
    }

    fn _param_mode(&self, param_nr: u32) -> i64 {
        let param_modes = self.memory[self.instr_ptr] / 100;
        param_modes / 10i64.pow(param_nr) % 10
    }

    fn _param_idx(&self, param_nr: usize) -> usize {
        self.instr_ptr + param_nr
    }
}

fn run_intcode_program(memory: Vec<i64>) -> i64 {
    let mut program = Program::from_memory(memory);
    program.run();
    program.memory[0]
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
    let input = puzzle_input!(3);

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
    let input = puzzle_input!(3);

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

// ===============================================================================================
//                                      Day 5
// ===============================================================================================

fn day_5(part: Part) {
    let puzzle_input = puzzle_input!(5);
    let input = match part {
        Part::One => 1,
        Part::Two => 5,
    };
    let mut output = Program::execute(puzzle_input, vec![input]);

    output.retain(|&code| code != 0);
    assert_eq!(output.len(), 1);
    println!("{}", output[0]);
}

#[test]
fn day_5_input_ouput() {
    for i in 0..10 {
        let output = Program::execute(
            "3,0,4,0,99",
            vec![i],
        );
        assert_eq!(i, output[0]);
    }
}


#[test]
fn day_5_position_mode_equal_to_8() {
    for i in 5..12 {
        let output = Program::execute(
            "3,9,8,9,10,9,4,9,99,-1,8",
            vec![i],
        );
        assert_eq!(output, vec![(i == 8) as _]);
    }
}
#[test]
fn day_5_immediate_mode_equal_to_8() {
    for i in 5..12 {
        let output = Program::execute(
            "3,3,1108,-1,8,3,4,3,99",
            vec![i],
        );
        assert_eq!(output, vec![(i == 8) as _]);
    }
}

#[test]
fn day_5_jump_test_position_mode() {
    for i in -5..5 {
        let output = Program::execute(
            "3,12,6,12,15,1,13,14,13,4,13,99,-1,0,1,9",
            vec![i],
        );
        assert_eq!(output, vec![(i != 0) as _]);
    }

}

#[test]
fn day_5_jump_test_immediate_mode() {
    for i in -5..5 {
        let output = Program::execute(
            "3,3,1105,-1,9,1101,0,0,12,4,12,99,1",
            vec![i],
        );
        assert_eq!(output, vec![(i != 0) as _]);
    }

}

#[test]
fn day_5_larger_example() {
    let puzzle_input = "3,21,1008,21,8,20,1005,20,22,107,8,21,20,1006,20,31,1106,0,36,98,0,0,1002,21,125,20,4,20,1105,1,46,104,999,1105,1,46,1101,1000,1,20,4,20,1105,1,46,98,99";

    for input in 5..12 {
        let output = Program::execute(puzzle_input, vec![input]);
        assert_eq!(output[0], 1000 - 8.cmp(&input) as i64)
    }
}


// ===============================================================================================
//                                      Day 6
// ===============================================================================================

fn day_6(part: Part) {
    // store x orbits y
    // or y is orbited by x?
    // different performance tradeoffs, but can't tell from this small example
    // what's necessary.
    // orbited-by relations make computing the checksum cheaper, so I went for that
    let input = puzzle_input!(6);

    let orbited_orbiter_pairs = input.lines()
        .map(|line| {
            let (orbited, orbiter) = line.split_at(line.find(')').expect(""));
            let orbiter = &orbiter[1..]; // remove the ')'
            (orbited, orbiter)
        });

    match part {
        Part::One => {
            let mut orbiters_of = HashMap::new();
            for (orbited, orbiter) in orbited_orbiter_pairs {
                orbiters_of.entry(orbited).or_insert(vec![]).push(orbiter);
            }
            println!("day 6 part 1: {}", orbital_checksum(&orbiters_of));
        }
        Part::Two => {
            let mut santas_orbit = "";
            let mut my_orbit = "";
            let mut orbit_neighbors = HashMap::new();

            // construct graph and find start and endpoint
            for (orbited, orbiter) in orbited_orbiter_pairs {
                // only need these two, no need to store the rest
                if orbiter == "YOU" {
                    my_orbit = orbited;
                }
                if orbiter == "SAN" {
                    santas_orbit = orbited;
                }

                orbit_neighbors.entry(orbited).or_insert(vec![]).push(orbiter);
                orbit_neighbors.entry(orbiter).or_insert(vec![]).push(orbited);
            }

            // dijkstra
            let mut orbited_by_distance_to_my_orbit = BinaryHeap::new();
            let mut seen_already = HashSet::new();

            let mut add_orbits = |heap: &mut BinaryHeap<_>, orbit, distance| {
                for &orbit in orbit_neighbors[orbit].iter() {
                    let is_new = seen_already.insert(orbit);
                    if is_new {
                        heap.push((distance + 1, orbit));
                    }
                }
            };
            add_orbits(&mut orbited_by_distance_to_my_orbit, my_orbit, 0);

            let distance = loop {
                let (distance, next_closest_orbit) = orbited_by_distance_to_my_orbit.pop().unwrap();
                if next_closest_orbit == santas_orbit {
                    break distance;
                }
                add_orbits(&mut orbited_by_distance_to_my_orbit, next_closest_orbit, distance);

            };
            println!("day 6 part 2: {}", distance);
        }
    }
}

// ===============================================================================================
//                                      Day 7
// ===============================================================================================

fn day_7(part: Part) {
    // use this trait to generate all permutations
    use permutohedron::LexicalPermutation;
    let program = Program::new(include_str!("day_7_input.txt"));

    // 5! = 120 permutations
    // trivial to bruteforce
    match part {
        Part::One => {
            let mut phases = [0, 1, 2, 3, 4];

            let max_output = (0..120).map(|_| {
                    let output = phases.iter().fold(0, |input, &phase| {
                        let mut program = program.clone();
                        program.input = vec![phase, input].into();
                        program.run();
                        program.output[0]
                    });
                    phases.next_permutation();
                    output
                })
                .max()
                .unwrap();

            println!("{}", max_output);
        }
        Part::Two => {
            let mut phases = [5, 6, 7, 8, 9];

            let mut max_output = 0;
            for _ in 0..120 {
                let mut programs = [program.clone(), program.clone(), program.clone(), program.clone(), program.clone()];
                for (&phase, program) in phases.iter().zip(programs.iter_mut()) {
                    program.input.push_back(phase);
                }
                //programs[0].input.push_back(0);

                let mut input_output = 0;
                loop {
                    let mut is_final_iteration = false;
                    for program in programs.iter_mut() {
                        program.input.push_back(input_output);
                        is_final_iteration |= program.run() == ProgramState::Finished;
                        input_output = program.output.pop().unwrap();
                    }
                    if is_final_iteration {
                        break;
                    }
                }
                max_output = std::cmp::max(max_output, input_output);
                phases.next_permutation();
            }

            println!("day 7 part 2: {}", max_output);
        }
    }
}

fn orbital_checksum(orbiters_of: &HashMap<&str, Vec<&str>>) -> u32 {
    _orbital_checksum("COM", 0, &orbiters_of)
}

// recursing function
// the total number of orbits for an n-long chain is the nth triangle number
fn _orbital_checksum(orbiter: &str, depth: u32, orbiters_of: &HashMap<&str, Vec<&str>>) -> u32 {
    depth + orbiters_of.get(orbiter)
        .map_or(0, |orbiters|
            orbiters
                .iter()
                .map(|orbiter| _orbital_checksum(orbiter, depth + 1, orbiters_of))
                .sum::<u32>()
        )
}


// ===============================================================================================
//                                      Day 8
// ===============================================================================================

// const PIXEL_OFF: u8 = 0;
const PIXEL_ON: u8 = 1;
const PIXEL_TRANSPARENT: u8 = 2;

fn day_8(part: Part) {
    let input = puzzle_input!(8);
    const WIDTH: usize = 25;
    const HEIGHT: usize = 6;
    const N_PIXELS_PER_LAYER: usize = WIDTH * HEIGHT;

    let layers: Vec<Vec<_>> = input
        .trim()
        .as_bytes()
        .chunks(N_PIXELS_PER_LAYER)
        .map(|chunk| {
            let mut chunk = chunk.to_vec();
            // convert ascii to integers, assuming valid input
            chunk.iter_mut().for_each(|num| *num -= b'0');
            chunk
        })
        .collect();

    assert!(layers.last().map_or(false, |layer| layer.len() == N_PIXELS_PER_LAYER));

    match part {
        Part::One => {
            let count_digit = |layer: &Vec<_>, digit: u8| layer.iter().filter(|&&dig| dig == digit).count();
            let fewest_0_layer = layers.iter()
                .min_by_key(|layer| count_digit(layer, 0))
                .unwrap();

            let solution = count_digit(fewest_0_layer, 1) * count_digit(fewest_0_layer, 2);
            println!("day 8 part 1: {}", solution);
        }
        Part::Two => {
            let image = layers
                .into_iter()
                .fold1(|mut upper_layer, lower_layer| {
                    upper_layer.iter_mut().zip(lower_layer)
                        .filter(|(&mut upper_cell, _)| upper_cell == PIXEL_TRANSPARENT)
                        .for_each(|(upper_cell, lower_cell)| *upper_cell = lower_cell);
                    upper_layer
                })
                .unwrap();

            let row_of_pixel = |n_pixel| n_pixel / WIDTH;
            let rows = image
                .into_iter()
                .enumerate()
                .group_by(|&(n_pixel, _color)| row_of_pixel(n_pixel));

            println!("day 8 part 2:");
            for (_, row) in rows.into_iter() {
                for (_, value) in row {
                    print!("{0}{0}", if value == PIXEL_ON { '█' } else { ' ' });
                }
                println!();
            }
        }
    }
}

// ===============================================================================================
//                                      Day 9
// ===============================================================================================

fn day_9(part: Part) {
    let input = puzzle_input!(9);
    let output = Program::execute(input, vec![part as i64]);
    assert_eq!(output.len(), 1);
    println!("day 9 part {}: {}", part as i64, output[0]);
}

#[test]
fn day_9_quine() {
    let output = Program::execute(
        "109,1,204,-1,1001,100,1,100,1008,100,16,101,1006,101,0,99",
        vec![]
    );
    assert_eq!(
        output, vec![109,1,204,-1,1001,100,1,100,1008,100,16,101,1006,101,0,99],
    );
}

#[test]
fn day_9_large_number() {
    let output = Program::execute(
        "1102,34915192,34915192,7,4,7,99,0",
        vec![]
    );
    assert!(output.into_iter().all(|num| num >= 10i64.pow(15)));
}

#[test]
fn day_9_output_large_number_as_is() {
    let output = Program::execute(
        "104,1125899906842624,99",
        vec![]
    );
    assert_eq!(output, vec![1125899906842624]);
}

// ===============================================================================================
//                                      Day 10
// ===============================================================================================

// Angle stored as a 2D vector with minimized integer coefficients.
#[derive(Copy, Clone, PartialEq, PartialOrd, Ord, Eq, Debug, Hash)]
struct Vec2D {
    x: i16,
    y: i16,
}

impl Vec2D {
    // x == y == 0 is forbidden
    fn reduced(x: i16, y: i16) -> Self {
        let gcd = num::integer::gcd(x, y);
        Vec2D {
            x: x / gcd,
            y: y / gcd,
        }
    }

    fn from_tuple((x, y): (i16, i16)) -> Self {
        Vec2D { x, y }
    }
}

impl std::ops::Add for Vec2D {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Vec2D {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

fn day_10(part: Part) {
    use ord_subset::OrdSubsetSliceExt;
    let input = puzzle_input!(10);
    // true, if asteroid is at position
    let asteroid_grid: Vec<bool> = input
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .map(|ch| ch == '#')
        .collect();

    let width = input.lines().next().unwrap().len() as i16;
    let height = input.lines().count();

    let idx = |x, y| usize::try_from(y * width + x).unwrap();

    let all_positions = (0..width as i16)
        .flat_map(|x| (0..height as i16).map(move |y| (x, y)));
    let asteroid_positions = all_positions.filter(|&(x, y)| asteroid_grid[idx(x,y)])
        .collect::<Vec<_>>();

    let (n_asteroids, best_position) = asteroid_positions
        .iter()
        .copied()
        .map(|pos| (n_asteroids_visible_from_pos(pos, asteroid_positions.iter().copied()), pos))
        .max()
        .unwrap();

    match part {
        Part::One => {
            println!("day 10 part 1: {} asteroids visible (position: {:?})", n_asteroids, best_position);
        }
        Part::Two => {
            // there are more asteroids than 200 so I can safely ignore multiple rotations
            let angles = angles_with_asteroids_visible(best_position, asteroid_positions.iter().copied());
            // and their angles
            let mut visible_asteroids = angles
                .into_iter()
                .map(|angle| {
                    let asteroid_position = closest_visible_asteroid_in_direction(&asteroid_grid, width, best_position, angle);
                    (asteroid_position, angle)
                })
                .collect::<Vec<_>>();

            // sort by angle
            visible_asteroids.ord_subset_sort_by_key(|(_, angle)| {
                use std::f64::consts::PI;
                // map direction vec to float of the angle, 0 for straight up and increasing clock-wise
                // wish I could put a picture here of the derivation
                (PI - f64::atan2(angle.x as f64, angle.y as f64)) % (2.0 * PI)
            });

            let (Vec2D { x, y }, _) = visible_asteroids[199];
            println!("day 10 part 2: {}", x * 100 + y);
        }
    }
}

fn n_asteroids_visible_from_pos(pos: (i16, i16), asteroid_positions: impl Iterator<Item = (i16, i16)>) -> usize {
    angles_with_asteroids_visible(pos, asteroid_positions).len()
}

fn angles_with_asteroids_visible(pos: (i16, i16), asteroid_positions: impl Iterator<Item = (i16, i16)>) -> HashSet<Vec2D> {
    let (x, y) = pos;
    asteroid_positions
        .filter(|&other_pos| pos != other_pos)
        .map(|(x_ast, y_ast)| Vec2D::reduced(x_ast - x, y_ast - y))
        .collect::<HashSet<_>>()
}

// only use with angles from `angles_with_asteroids_visible`
// or it will panic
fn closest_visible_asteroid_in_direction(
    asteroid_grid: &[bool],
    width: i16,
    pos: (i16, i16),
    angle: Vec2D,
) -> Vec2D {
    let pos = Vec2D::from_tuple(pos);
    std::iter::successors(Some(pos), |&old_pos| Some(old_pos + angle))
        .skip(1) // don't count station itself
        .find(|pos| asteroid_grid[(width * pos.y + pos.x) as usize])
        .unwrap()
}

// code for printing asteroid grid with amount of visible asteroids
// marked
//
// for y in 0..height as i16 {
//     for x in 0..width as i16 {
//         if asteroid_grid[idx(x, y)] {
//             print!("{}", n_asteroids_visible_from_pos((x, y), asteroid_positions.clone()));
//         } else {
//             print!(".");
//         }
//     }
//     println!();
// }
#[bench]
fn day_10_part1(b: &mut test::Bencher) {
    b.iter(|| day_10(Part::One));
}

#[bench]
fn day_10_part2(b: &mut test::Bencher) {
    b.iter(|| day_10(Part::Two));
}

// ===============================================================================================
//                                      Day 11
// ===============================================================================================
const TURN_LEFT: i64 = 0;
const TURN_RIGHT: i64 = 1;

const BLACK: u8 = 0;
const WHITE: u8 = 1;

const PANEL_SIDELENGTH: usize = 100;

const LEFT: Vec2D = Vec2D { x: -1, y: 0 };
const RIGHT: Vec2D = Vec2D { x: 1, y: 0 };
const UP: Vec2D = Vec2D { x: 0, y: -1 };
const DOWN: Vec2D = Vec2D { x: 0, y: 1 };

fn day_11(part: Part) {
    // .....> x
    // |
    // |
    // v  y

    let middle = PANEL_SIDELENGTH as i16 / 2;
    let mut pos = Vec2D { x: middle, y: middle };
    let mut direction = Vec2D { x: 0, y: -1 };

    let turn_left = |direction| match direction {
        UP => LEFT,
        LEFT => DOWN,
        DOWN => RIGHT,
        RIGHT => UP,
        _ => unreachable!(),
    };

    // row major
    let mut panel_color = vec![[BLACK; PANEL_SIDELENGTH]; PANEL_SIDELENGTH];

    if part == Part::Two {
        panel_color[pos.y as usize][pos.x as usize] = WHITE;
    }

    let mut program = Program::new(include_str!("day_11_input.txt"));

    let mut panels_painted = HashSet::new();
    loop {
        let current_color = panel_color[pos.y as usize][pos.x as usize];
        program.input.push_back(current_color as _);
        let program_state = program.run();

        let new_color = program.output[0];
        direction = match program.output[1] {
            TURN_LEFT => turn_left(direction),
            // 3 lefts is 1 right
            TURN_RIGHT => turn_left(turn_left(turn_left(direction))),
            _ => unreachable!(),
        };
        program.output.clear();

        // puzzle is unclear on what counts as painting.
        // is black over black painting? Easier to count it as yes.
        panels_painted.insert(pos);
        panel_color[pos.y as usize][pos.x as usize] = new_color as _;
        pos = pos + direction;

        if program_state == ProgramState::Finished {
            break;
        }
    }

    match part {
        Part::One => println!("{}", panels_painted.len()),
        Part::Two => {
            let printed_panel = panel_color
                .into_iter()
                .map(|row| row.iter()
                    .map(|&color| if color == WHITE { "██"} else { "  " })
                    .collect::<String>()
                )
                .filter(|s| !s.trim().is_empty())
                .join("\n");
            let printed_panel = textwrap::dedent(&printed_panel);
            println!("{}", printed_panel);
        }
    }
}
// ===============================================================================================
//                                      Day 12
// ===============================================================================================
const DAY_12_SIMULATION_STEPS: u32 = 1000;
type INT = i16;

fn day_12(part: Part) {
    let puzzle_input = puzzle_input!(12);
    let pattern = regex::Regex::new(r"<x=(-?\d+), y=(-?\d+), z=(-?\d+)>").unwrap();

    let mut moon_positions = [vec![], vec![], vec![]];
    for line in puzzle_input.lines() {
        let captures = pattern.captures(line).unwrap();
        for axis in 0..3 {
            let coord = captures[axis+1].parse::<INT>().unwrap();
            moon_positions[axis].push(coord);
        }
    }
    let n_moons = moon_positions[0].len();
    assert_eq!(n_moons, 4);
    let mut moon_velocities = [
        vec![0; n_moons],
        vec![0; n_moons],
        vec![0; n_moons],
    ];

    match part {
        Part::One => {
            for axis in 0..3 {
                for _ in 0..DAY_12_SIMULATION_STEPS {
                    simulate_step(&mut moon_positions[axis], &mut moon_velocities[axis]);
                }
            }
            let vector_energy = |vector_collection: &[Vec<INT>; 3], n_moon| {
                vector_collection
                    .iter()
                    .map(|axis_collection| axis_collection[n_moon])
                    .map(INT::abs)
                    .sum::<INT>()
            };

            let total_energy = (0..n_moons)
                .map(|n_moon| vector_energy(&moon_positions, n_moon) * vector_energy(&moon_velocities, n_moon))
                .sum::<INT>();

            println!("day 12 part 1: {}", total_energy);
        }
        Part::Two => {
            // find the cycle length of each axis separately
            let cycle_lengths = (0..3)
                .map(|axis| {
                    let axis_positions = &mut moon_positions[axis];
                    let axis_velocities = &mut moon_velocities[axis];
                    let initial_positions = axis_positions.clone();
                    let initial_velocities = vec![0; n_moons];

                    // compute cycle length
                    // the first state that the system will encounter
                    // that it has already seen, will be the very first state.
                    // Classic result from statistical mechanics. The phasespace path
                    // will never cross itself, because
                    // 1. the system's behaviour is determined purely from its current state
                    // 2. it's time-inversion symmetric
                    // Thus there can only be at most two paths going into and out of every point.
                    // One forward in time, one backward.
                    // Cycles aren't guaranteed though.
                    (1..)
                        .find(|_| {
                            simulate_step(axis_positions, axis_velocities);
                            *axis_positions == initial_positions && *axis_velocities == initial_velocities
                        })
                        .unwrap()
                })
                .collect::<Vec<i64>>();

            use num::integer::lcm;
            // lcm(a,b,c) == lcm(lcm(a,b), c)
            let c = cycle_lengths;
            let total_cycle_length = lcm(lcm(c[0], c[1]), c[2]);
            println!("day 12 part 2: {}", total_cycle_length);
        }
    }
}

fn simulate_step(axis_positions: &mut [i16], axis_velocities: &mut [i16]) {
    for (&moon1_pos, moon1_vel) in axis_positions.iter().zip(&mut axis_velocities[..]) {
        for &moon2_pos in axis_positions.iter() {
                *moon1_vel += (moon2_pos - moon1_pos).signum();
        }
    }

    for (moon_pos, &moon_velocity) in axis_positions.iter_mut().zip(axis_velocities.iter()) {
        *moon_pos += moon_velocity;
    }
}

#[bench]
fn day_12_part1(b: &mut test::Bencher) {
    b.iter(|| day_12(Part::One));
}

#[bench]
fn day_12_part2(b: &mut test::Bencher) {
    b.iter(|| day_12(Part::Two));
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
        day_4(Part::Two);
        day_5(Part::One);
        day_5(Part::Two);
        day_6(Part::One);
        day_6(Part::Two);
        day_7(Part::One);
        day_7(Part::Two);
        day_8(Part::One);
        day_8(Part::Two);
        day_9(Part::One);
        day_9(Part::Two);
        day_10(Part::One);
        day_10(Part::Two);
        day_11(Part::One);
        day_11(Part::Two);
        day_12(Part::One);
    }
    day_12(Part::Two);
}
