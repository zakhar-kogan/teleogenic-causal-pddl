# Copyright 2022 Technion project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unified_planning as up
from unified_planning.shortcuts import *
from unified_planning.test import TestCase, main
import up_social_laws
from up_social_laws.single_agent_projection import SingleAgentProjection
from up_social_laws.robustness_verification import RobustnessVerifier, SimpleInstantaneousActionRobustnessVerifier, WaitingActionRobustnessVerifier
from up_social_laws.robustness_checker import SocialLawRobustnessChecker, SocialLawRobustnessStatus
from up_social_laws.social_law import SocialLaw
from up_social_laws.waitfor_specification import WaitforSpecification
from up_social_laws.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from up_social_laws.ma_centralizer import MultiAgentProblemCentralizer
from up_social_laws.sa_to_ma_converter import SingleAgentToMultiAgentConverter
from up_social_laws.synthesis import SocialLawGenerator, SocialLawGeneratorSearch, get_gbfs_social_law_generator
from unified_planning.model.multi_agent import *
from unified_planning.io import PDDLWriter, PDDLReader
from unified_planning.engines import PlanGenerationResultStatus
from collections import namedtuple
import random
import os

POSITIVE_OUTCOMES = frozenset(
    [
        PlanGenerationResultStatus.SOLVED_SATISFICING,
        PlanGenerationResultStatus.SOLVED_OPTIMALLY,
    ]
)

UNSOLVABLE_OUTCOMES = frozenset(
    [
        PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY,
        PlanGenerationResultStatus.UNSOLVABLE_PROVEN,
    ]
)

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
PDDL_DOMAINS_PATH = os.path.join(FILE_PATH, "pddl")

Example = namedtuple("Example", ["problem", "plan"])

class RobustnessTestCase:
    def __init__(self, name, 
                    expected_outcome : SocialLawRobustnessStatus, 
                    cars = ["car-north", "car-south", "car-east", "car-west"], 
                    yields_list = [], 
                    wait_drive = True):
        self.name = name
        self.cars = cars
        self.yields_list = yields_list
        self.expected_outcome = expected_outcome
        self.wait_drive = wait_drive        


def get_intersection_problem(
    cars = ["car-north", "car-south", "car-east", "car-west"], 
    yields_list = [], 
    wait_drive = True,
    durative = False) -> MultiAgentProblemWithWaitfor:
    # intersection multi agent
    problem = MultiAgentProblemWithWaitfor("intersection")

    loc = UserType("loc")
    direction = UserType("direction")
    car = UserType("car")

    # Environment     
    connected = Fluent('connected', BoolType(), l1=loc, l2=loc, d=direction)
    free = Fluent('free', BoolType(), l=loc)
    if len(yields_list) > 0:
        yieldsto = Fluent('yieldsto', BoolType(), l1=loc, l2=loc)
        problem.ma_environment.add_fluent(yieldsto, default_initial_value=False)
        dummy_loc = unified_planning.model.Object("dummy", loc)
        problem.add_object(dummy_loc)
    
    problem.ma_environment.add_fluent(connected, default_initial_value=False)
    problem.ma_environment.add_fluent(free, default_initial_value=True)

    intersection_map = {
        "north": ["south-ent", "cross-se", "cross-ne", "north-ex"],
        "south": ["north-ent", "cross-nw", "cross-sw", "south-ex"],
        "west": ["east-ent", "cross-ne", "cross-nw", "west-ex"],
        "east": ["west-ent", "cross-sw", "cross-se", "east-ex"]
    }

    location_names = set()
    for l in intersection_map.values():
        location_names = location_names.union(l)
    locations = list(map(lambda l: unified_planning.model.Object(l, loc), location_names))
    problem.add_objects(locations)

    direction_names = intersection_map.keys()
    directions = list(map(lambda d: unified_planning.model.Object(d, direction), direction_names))
    problem.add_objects(directions)

    for d, l in intersection_map.items():
        for i in range(len(l)-1):            
            problem.set_initial_value(connected(unified_planning.model.Object(l[i], loc), unified_planning.model.Object(l[i+1], loc), unified_planning.model.Object(d, direction)), True)

    # Agents
    at = Fluent('at', BoolType(), l1=loc)    
    arrived = Fluent('arrived', BoolType())
    not_arrived = Fluent('not-arrived', BoolType())
    start = Fluent('start', BoolType(), l=loc)        
    traveldirection = Fluent('traveldirection', BoolType(), d=direction)
    
    #  (:action arrive
        #     :agent    ?a - car 
        #     :parameters  (?l - loc)
        #     :precondition  (and  
        #     	(start ?a ?l)
        #     	(not (arrived ?a))
        #     	(free ?l)      
        #       )
        #     :effect    (and     	
        #     	(at ?a ?l)
        #     	(not (free ?l))
        #     	(arrived ?a)
        #       )
        #   )
    if durative:
        arrive = DurativeAction('arrive', l=loc)
        arrive.set_fixed_duration(1)        
        l = arrive.parameter('l')
        
        arrive.add_condition(StartTiming(),start(l))
        arrive.add_condition(StartTiming(),not_arrived())
        arrive.add_condition(OpenTimeInterval(StartTiming(), EndTiming()),free(l))
        arrive.add_effect(EndTiming(), at(l), True)
        arrive.add_effect(EndTiming(), free(l), False)
        arrive.add_effect(EndTiming(), arrived(), True)        
        arrive.add_effect(EndTiming(), not_arrived(), False)        
    else:
        arrive = InstantaneousAction('arrive', l=loc)    
        l = arrive.parameter('l')
        arrive.add_precondition(start(l))
        arrive.add_precondition(not_arrived())
        arrive.add_precondition(free(l))
        arrive.add_effect(at(l), True)
        arrive.add_effect(free(l), False)
        arrive.add_effect(arrived(), True)   
        arrive.add_effect(not_arrived(), False)   



    #   (:action drive
    #     :agent    ?a - car 
    #     :parameters  (?l1 - loc ?l2 - loc ?d - direction ?ly - loc)
    #     :precondition  (and      	
    #     	(at ?a ?l1)
    #     	(free ?l2)     
    #     	(travel-direction ?a ?d)
    #     	(connected ?l1 ?l2 ?d)
    #     	(yields-to ?l1 ?ly)
    #     	(free ?ly)
    #       )
    #     :effect    (and     	
    #     	(at ?a ?l2)
    #     	(not (free ?l2))
    #     	(not (at ?a ?l1))
    #     	(free ?l1)
    #       )
    #    )    
    # )
    if durative:
        if len(yields_list) > 0:
            drive = DurativeAction('drive', l1=loc, l2=loc, d=direction, ly=loc)
        else:
            drive = DurativeAction('drive', l1=loc, l2=loc, d=direction)
        drive.set_fixed_duration(1)        
        l1 = drive.parameter('l1')
        l2 = drive.parameter('l2')
        d = drive.parameter('d')        
        drive.add_condition(StartTiming(), at(l1))
        if wait_drive:        
            drive.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), free(l2))
        drive.add_condition(StartTiming(), traveldirection(d))
        drive.add_condition(EndTiming(), connected(l1,l2,d))        
        drive.add_effect(EndTiming(), at(l2),True)
        drive.add_effect(EndTiming(), free(l2), False)
        drive.add_effect(StartTiming(), at(l1), False)
        drive.add_effect(EndTiming(), free(l1), True)
        if len(yields_list) > 0:
            ly = drive.parameter('ly')
            drive.add_condition(StartTiming(), yieldsto(l1,ly))
            drive.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), free(ly))

    else:
        if len(yields_list) > 0:
            drive = InstantaneousAction('drive', l1=loc, l2=loc, d=direction, ly=loc)    
        else:
            drive = InstantaneousAction('drive', l1=loc, l2=loc, d=direction)
        l1 = drive.parameter('l1')
        l2 = drive.parameter('l2')
        d = drive.parameter('d')
        #ly = drive.parameter('ly')
        drive.add_precondition(at(l1))
        drive.add_precondition(free(l2))  # Remove for yield/wait
        drive.add_precondition(traveldirection(d))
        drive.add_precondition(connected(l1,l2,d))
        if len(yields_list) > 0:
            ly = drive.parameter('ly')
            drive.add_precondition(yieldsto(l1,ly))
            drive.add_precondition(free(ly))
        drive.add_effect(at(l2),True)
        drive.add_effect(free(l2), False)
        drive.add_effect(at(l1), False)
        drive.add_effect(free(l1), True)    



    plan = up.plans.SequentialPlan([])

    for d, l in intersection_map.items():
        carname = "car-" + d
        if carname in cars:
            car = Agent(carname, problem)
        
            problem.add_agent(car)
            car.add_fluent(at, default_initial_value=False)
            car.add_fluent(arrived, default_initial_value=False)
            car.add_fluent(not_arrived, default_initial_value=True)
            car.add_fluent(start, default_initial_value=False)
            car.add_fluent(traveldirection, default_initial_value=False)
            car.add_action(arrive)
            car.add_action(drive)

            slname = l[0]
            slobj = unified_planning.model.Object(slname, loc)

            glname = l[-1]
            globj = unified_planning.model.Object(glname, loc)
            
            dobj = unified_planning.model.Object(d, direction)

            problem.set_initial_value(Dot(car, car.fluent("start")(slobj)), True)
            problem.set_initial_value(Dot(car, car.fluent("traveldirection")(dobj)), True)        
            car.add_public_goal(car.fluent("at")(globj))
            #problem.add_goal(Dot(car, car.fluent("at")(globj)))

            if len(yields_list) > 0:
                yields = set()
                for l1_name, ly_name in yields_list:
                    problem.set_initial_value(yieldsto(problem.object(l1_name), problem.object(ly_name)), True)     
                    yields.add(problem.object(l1_name))
                for l1 in problem.objects(loc):
                    if l1 not in yields:
                        problem.set_initial_value(yieldsto(l1, dummy_loc), True)        

            # slobjexp1 = (ObjectExp(slobj)),        
            # plan.actions.append(up.plans.ActionInstance(arrive, slobjexp1, car))

            # for i in range(1,len(l)):
            #     flname = l[i-1]
            #     tlname = l[i]
            #     flobj = unified_planning.model.Object(flname, loc)
            #     tlobj = unified_planning.model.Object(tlname, loc)
            #     plan.actions.append(up.plans.ActionInstance(drive, (ObjectExp(flobj), ObjectExp(tlobj), ObjectExp(dobj) ), car))

    # Add waitfor annotations
    for agent in problem.agents:
        drive = agent.action("drive")
        l2 = drive.parameter("l2")        
        if wait_drive:
            problem.waitfor.annotate_as_waitfor(agent.name, drive.name, free(l2))
        if len(yields_list) > 0:
            ly = drive.parameter("ly")
            problem.waitfor.annotate_as_waitfor(agent.name, drive.name, free(ly)) 



    intersection = Example(problem=problem, plan=plan)
    return intersection


class TestProblem(TestCase):
    def setUp(self):
        TestCase.setUp(self)        
        self.test_cases = [         
            RobustnessTestCase("4cars_crash", SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_FAIL, yields_list=[], wait_drive=False),   
            RobustnessTestCase("4cars_deadlock", SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_DEADLOCK, yields_list=[]),
            RobustnessTestCase("4cars_yield_deadlock", SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_DEADLOCK, yields_list=[("south-ent", "east-ent"),("east-ent", "north-ent"),("north-ent", "west-ent"),("west-ent", "south-ent")]),
            RobustnessTestCase("4cars_robust", SocialLawRobustnessStatus.ROBUST_RATIONAL, yields_list=[("south-ent", "cross-ne"),("north-ent", "cross-sw"),("east-ent", "cross-nw"),("west-ent", "cross-se")]),
            RobustnessTestCase("2cars_crash", SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_FAIL, cars=["car-north", "car-east"], yields_list=[], wait_drive=False),   
            RobustnessTestCase("2cars_robust", SocialLawRobustnessStatus.ROBUST_RATIONAL, cars=["car-north", "car-south"], yields_list=[], wait_drive=False)            
        ]

    def test_synthesis(self):
        problem = MultiAgentProblemWithWaitfor()
        
        loc = UserType("loc")
    
        # Environment     
        connected = Fluent('connected', BoolType(), l1=loc, l2=loc)        
        problem.ma_environment.add_fluent(connected, default_initial_value=False)

        free = Fluent('free', BoolType(), l=loc)
        problem.ma_environment.add_fluent(free, default_initial_value=True)

        nw, ne, sw, se = Object("nw", loc), Object("ne", loc), Object("sw", loc), Object("se", loc)        
        problem.add_objects([nw, ne, sw, se])
        problem.set_initial_value(connected(nw, ne), True)
        problem.set_initial_value(connected(nw, sw), True)
        problem.set_initial_value(connected(ne, nw), True)
        problem.set_initial_value(connected(ne, se), True)
        problem.set_initial_value(connected(sw, se), True)
        problem.set_initial_value(connected(sw, nw), True)
        problem.set_initial_value(connected(se, sw), True)
        problem.set_initial_value(connected(se, ne), True)


        at = Fluent('at', BoolType(), l1=loc)

        move = InstantaneousAction('move', l1=loc, l2=loc)
        l1 = move.parameter('l1')
        l2 = move.parameter('l2')
        move.add_precondition(at(l1))
        move.add_precondition(free(l2))
        move.add_precondition(connected(l1,l2))
        move.add_effect(at(l2),True)
        move.add_effect(free(l2), False)
        move.add_effect(at(l1), False)
        move.add_effect(free(l1), True)    

        agent1 = Agent("a1", problem)
        problem.add_agent(agent1)
        agent1.add_fluent(at, default_initial_value=False)
        agent1.add_action(move)
        problem.waitfor.annotate_as_waitfor(agent1.name, move.name, free(l2))

        agent2 = Agent("a2", problem)
        problem.add_agent(agent2)
        agent2.add_fluent(at, default_initial_value=False)
        agent2.add_action(move)
        problem.waitfor.annotate_as_waitfor(agent2.name, move.name, free(l2))

        problem.set_initial_value(Dot(agent1, at(nw)), True)
        problem.set_initial_value(Dot(agent2, at(se)), True)
        problem.set_initial_value(free(nw), False)
        problem.set_initial_value(free(se), False)

        agent1.add_public_goal(at(sw))
        agent2.add_public_goal(at(ne))
        # problem.add_goal(Dot(agent1, at(sw)))
        # problem.add_goal(Dot(agent2, at(ne)))


        slrc = SocialLawRobustnessChecker(
            planner_name="fast-downward",
            robustness_verifier_name="SimpleInstantaneousActionRobustnessVerifier",
            save_pddl_prefix="synth"
            )
        l = SocialLaw()
        l.disallow_action("a1", "move", ("nw","ne"))
        l.disallow_action("a1", "move", ("sw","se"))
        l.disallow_action("a2", "move", ("ne","nw"))
        l.disallow_action("a2", "move", ("se","sw"))
        pr = l.compile(problem).problem
        # prr = slrc.is_robust(pr)
        # self.assertEqual(prr.status,SocialLawRobustnessStatus.ROBUST_RATIONAL)

        l2 = SocialLaw()
        l.disallow_action("a1", "move", ("nw","ne"))
        l.disallow_action("a2", "move", ("ne","nw"))

        self.assertTrue(l.is_stricter_than(l2))
        self.assertFalse(l2.is_stricter_than(l))

        # g1 = SocialLawGenerator(SocialLawGeneratorSearch.BFS)
        # rprob1 = g1.generate_social_law(problem)
        # self.assertIsNotNone(rprob1)

        # g2 = SocialLawGenerator(SocialLawGeneratorSearch.DFS)
        # rprob2 = g2.generate_social_law(problem)
        # self.assertIsNotNone(rprob2)

        g3 = get_gbfs_social_law_generator()
        rprob3 = g3.generate_social_law(problem)
        self.assertIsNotNone(rprob3)
        

    def test_social_law(self):
        slrc = SocialLawRobustnessChecker(
            planner_name="fast-downward",
            robustness_verifier_name="SimpleInstantaneousActionRobustnessVerifier"
            )
        p_4cars_crash = get_intersection_problem(wait_drive=False).problem
        l = SocialLaw()
        for agent in p_4cars_crash.agents:
            l.add_waitfor_annotation(agent.name, "drive", "free", ("l2",)  )
        
        res = l.compile(p_4cars_crash)
        p_4cars_deadlock = res.problem
        self.assertEqual(len(p_4cars_crash.waitfor.waitfor_map), 0)
        self.assertEqual(len(p_4cars_deadlock.waitfor.waitfor_map), 4)

        r_result = slrc.is_robust(p_4cars_deadlock)
        self.assertEqual(r_result.status, SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_DEADLOCK)
        r_result = slrc.is_robust(p_4cars_crash)
        self.assertEqual(r_result.status, SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_FAIL)        

        l2 = SocialLaw()
        l2.disallow_action("car-north", "drive", ("south-ent", "cross-se", "north") )
        res = l2.compile(p_4cars_crash)
        p_nosap = res.problem

        r_result = slrc.is_robust(p_nosap)
        self.assertEqual(r_result.status, SocialLawRobustnessStatus.NON_ROBUST_SINGLE_AGENT)

        l3 = SocialLaw()        
        l3.add_new_fluent(None, "yieldsto", (("l1","loc"), ("l2","loc")), False)
        l3.add_new_object("dummy_loc", "loc")
        for loc1,loc2 in [("south-ent", "cross-ne"),("north-ent", "cross-sw"),("east-ent", "cross-nw"),("west-ent", "cross-se")]:
            l3.set_initial_value_for_new_fluent(None, "yieldsto", (loc1, loc2), True)
        for loc in p_4cars_crash.objects(p_4cars_crash.user_type("loc")):
            if loc.name not in ["south-ent", "north-ent", "east-ent", "west-ent"]:
                l3.set_initial_value_for_new_fluent(None, "yieldsto", (loc.name, "dummy_loc"), True)
        for agent in p_4cars_crash.agents:
            l3.add_parameter_to_action(agent.name, "drive", "ly", "loc")            
            l3.add_precondition_to_action(agent.name, "drive", "yieldsto", ("l1", "ly") )            
            l3.add_precondition_to_action(agent.name, "drive", "free", ("ly",) )
            l3.add_waitfor_annotation(agent.name, "drive", "free", ("ly",) )
        res = l3.compile(p_4cars_deadlock)
        p_robust = res.problem
        r_result = slrc.is_robust(p_robust)
        self.assertEqual(r_result.status, SocialLawRobustnessStatus.ROBUST_RATIONAL)
        self.assertEqual(len(p_robust.ma_environment.fluents), len(p_4cars_deadlock.ma_environment.fluents) + 1)

    def test_all_cases(self):
        for t in self.test_cases:
            problem = get_intersection_problem(t.cars, t.yields_list, t.wait_drive, durative=False).problem
            slrc = SocialLawRobustnessChecker(
                planner_name="fast-downward",
                robustness_verifier_name="SimpleInstantaneousActionRobustnessVerifier"
                )
            r_result = slrc.is_robust(problem)
            self.assertEqual(r_result.status, t.expected_outcome, t.name)
            if t.expected_outcome == SocialLawRobustnessStatus.ROBUST_RATIONAL:
                presult = slrc.solve(problem)
                self.assertIn(presult.status, POSITIVE_OUTCOMES, t.name)

    # def test_all_cases_waiting(self):
    #     for t in self.test_cases:
    #         problem = get_intersection_problem(t.cars, t.yields_list, t.wait_drive, durative=False).problem
    #         slrc = SocialLawRobustnessChecker(
    #             planner_name="fast-downward",
    #             robustness_verifier_name="WaitingActionRobustnessVerifier"
    #             )
    #         r_result = slrc.is_robust(problem)
    #         self.assertEqual(r_result.status, t.expected_outcome, t.name)
    #         if t.expected_outcome == SocialLawRobustnessStatus.ROBUST_RATIONAL:
    #             presult = slrc.solve(problem)
    #             self.assertIn(presult.status, POSITIVE_OUTCOMES, t.name)


    def test_centralizer(self):
        for t in self.test_cases:
            for durative in [False]:# True]:
                problem = get_intersection_problem(t.cars, t.yields_list, t.wait_drive, durative=durative).problem
                mac = MultiAgentProblemCentralizer()
                cresult = mac.compile(problem)
                with OneshotPlanner(problem_kind=cresult.problem.kind) as planner:
                    presult = planner.solve(cresult.problem)
                    self.assertIn(presult.status, POSITIVE_OUTCOMES, t.name)
 
        
    def test_all_cases_durative(self):
        for t in self.test_cases:
            problem = get_intersection_problem(t.cars, t.yields_list, t.wait_drive, durative=True).problem
            with open("waitfor.json", "w") as f:
                f.write(str(problem.waitfor))

            slrc = SocialLawRobustnessChecker(                                
                save_pddl_prefix=t.name
                )
            self.assertEqual(slrc.is_robust(problem).status, t.expected_outcome, t.name)

    def test_sa_ma_converter(self):
        reader = PDDLReader()
        random.seed(2023)
        
        domain_filename = os.path.join(PDDL_DOMAINS_PATH, "transport", "domain.pddl")
        problem_filename = os.path.join(PDDL_DOMAINS_PATH, "transport", "task10.pddl")
        problem = reader.parse_problem(domain_filename, problem_filename)

        samac = SingleAgentToMultiAgentConverter(["vehicle"])

        result = samac.compile(problem)

        print(result.problem)