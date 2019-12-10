use itertools::Itertools;
use std::collections::VecDeque;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::BinaryHeap;

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
    let input = include_str!("day_2_input.txt");
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

#[derive(Clone)]
struct Program {
    memory: Vec<i64>,
    instr_ptr: usize,
    input: VecDeque<i64>,
    output: Vec<i64>,
}

const POSITION_MODE: i64 = 0;
const IMMEDIATE_MODE: i64 = 1;

impl Program {
    fn new(serialized_memory: &str) -> Self {
        Self::from_memory(Self::parse_memory(serialized_memory))
    }

    fn from_memory(memory: Vec<i64>) -> Self {
        Self {
            memory,
            instr_ptr: 0,
            input: VecDeque::new(),
            output: vec![],
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

    fn run(&mut self) -> ProgramState {
        loop {
            let instruction = self.memory[self.instr_ptr];
            let opcode = instruction % 100;

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
                    let result_pos = self.return_addr(2);

                    self.memory[result_pos] = op(val1, val2);
                    self.instr_ptr += 4;
                }
                // input
                3 => {
                    let return_pos = self.return_addr(0);
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
                    let return_pos = self.return_addr(2);
                    self.memory[return_pos] = comparator(&self.param_val(0), &self.param_val(1)) as _;
                    self.instr_ptr += 4;
                }
                99 => break,
                _ => unreachable!(),
            }

            assert_ne!(_old_instr_ptr, self.instr_ptr);
        }
        ProgramState::Finished
    }

    fn param_val(&self, param_nr: u32) -> i64 {
        let param_modes = self.memory[self.instr_ptr] / 100;
        let param_mode = param_modes / 10i64.pow(param_nr) % 10;
        self._param_val(param_nr, param_mode)
    }

    fn _param_val(&self, param_nr: u32, param_mode: i64) -> i64 {
        let param_offset = param_nr as usize + 1;
        let address = match param_mode {
            POSITION_MODE => self.memory[self._param_idx(param_offset)] as usize,
            IMMEDIATE_MODE => self._param_idx(param_offset),
            _ => unreachable!(),
        };
        self.memory[address]
    }

    fn return_addr(&self, param_nr: u32) -> usize {
        // resolution will be done by caller when assigning to it
        self._param_val(param_nr, IMMEDIATE_MODE) as usize
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
    let input = include_str!("day_3_input.txt");

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
    let input = include_str!("day_3_input.txt");

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
    let puzzle_input = include_str!("day_5_input.txt");
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
    let input = include_str!("day_6_input.txt");

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
    let input = include_str!("day_8_input.txt");
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
    }
    day_8(Part::Two);
}
