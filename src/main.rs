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

// in manhattan distance, a square is a circle
const GRID_RADIUS: i32 = 10000;

fn day_3(_part: Part) {
    let input: &str = include_str!("day_3_input.txt");
    let n_lines = input.lines().count();

    // necessary size guessed, will crash if it's too little
    // middle at (GRID_RADIUS, GRID_RADIUS)
    // with GRID_RADIUS cells to each side
    let mut wires_on_cell = vec![[0u8; 2*GRID_RADIUS as usize + 1]; 2*GRID_RADIUS as usize + 1];

    for (wire_nr, wire) in input.lines().enumerate() {
        assert!(wire_nr < n_lines);
        let wire_mask = 1 << wire_nr;

        let mut x = GRID_RADIUS;
        let mut y = GRID_RADIUS;
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
                wires_on_cell[y as usize][x as usize] |= wire_mask;
            }
        }
    }

    // inefficient, but simple
    let distance = wires_on_cell.iter().enumerate()
        .flat_map(|(y, row)| row.iter().enumerate().map(move |(x, wires)| (x, y, wires.count_ones())))
        .filter(|&(_, _, n_wires)| n_wires > 1)
        .map(|(x, y, _)| (x as i32 - GRID_RADIUS).abs() + (y as i32 - GRID_RADIUS).abs())
        .min()
        .unwrap();
    println!("{}", distance);
}

fn main() {
    // keep old code in here to avoid unused function warnings
    if false {
        day_1(Part::One);
        day_1(Part::Two);
        day_2(Part::One);
        day_2(Part::Two);
    }

    day_3(Part::One);
}
