from fst import *

# here are some predefined character sets that might come in handy.
# you can define your own
AZ = set("abcdefghijklmnopqrstuvwxyz")
VOWS = set("aeiou")
N = set("n")
P = set("p")
T = set("t")
R = set("r")
NPTR = set("nptr")
CONS = AZ-VOWS
E = set("e")
U = set("u")
I = set("i")


# Implement your solution here
def buildFST():
    print("Your task is to implement a better FST in the buildFST() function, using the methods described here")
    print("You may define additional methods in this module (hw1_fst.py) as desired")
    #
    # The states (you need to add more)
    # ---------------------------------------
    # 
    f = FST("s") # q0 is the initial (non-accepting) state
    f.addState("x") # a non-accepting state
    f.addState("y") # a non-accepting state
    f.addState("z")
    f.addState("ing")

    f.addState("ne")
    f.addState("ne_1")
    f.addState("ne_2")


    f.addState("q3")
    f.addState("q3_1")
    f.addState("q1")

    f.addState("q2")
    f.addState("q2_n")
    f.addState("q2_p")
    f.addState("q2_t")
    f.addState("q2_r")
    f.addState("q2_e")




    f.addState("EOW", True) # an accepting state (you shouldn't need any additional accepting states)

    # c/o from s
    f.addSetTransition("s", AZ, "s")
    f.addSetTransition("s", AZ-NPTR-E, "ing")
    f.addSetTransition("s", CONS, "ne_2")
    f.addSetTransition("s", VOWS, "ne_1")

    f.addSetTransition("ne_1", VOWS, "ne_2")

    f.addSetTransition("ne_2", NPTR, "ing")


    # rule 1
    f.addSetTransition("s", CONS, "q1")
    f.addSetTransition("s", U, "q1")
    f.addTransition("q1","e","","ing")
    
    # rule 2
    f.addSetTransition("s", CONS, "x")

    # c/o from x
    f.addSetTransition("x", VOWS-E, "q2")
    f.addSetTransition("x", E, "q2_e")

    # c/o from q2_e
    f.addSetTransition("q2_e", P, "q2_p")
    f.addSetTransition("q2_e", T, "q2_t")
    f.addSetTransition("q2_e", R, "ing")
    f.addSetTransition("q2_e", N, "ing")
    f.addSetTransition("q2_e", E, "ing")

    # c/o from q2
    f.addSetTransition("q2", N, "q2_n")
    f.addSetTransition("q2", P, "q2_p")
    f.addSetTransition("q2", T, "q2_t")
    f.addSetTransition("q2", R, "q2_r")

    f.addTransition("q2_n","","n","ing")
    f.addTransition("q2_p","","p","ing")
    f.addTransition("q2_t","","t","ing")
    f.addTransition("q2_r","","r","ing")

    # rule 3
    f.addTransition("s","i","","q3")
    f.addTransition("q3","e","y","ing")

    # final adding ing
    f.addTransition("ing", "", "ing", "EOW")

    # Return your completed FST
    return f
    

if __name__ == "__main__":
    # Pass in the input file as an argument
    if len(sys.argv) < 2:
        print("This script must be given the name of a file containing verbs as an argument")
        quit()
    else:
        file = sys.argv[1]
    #endif

    # Construct an FST for translating verb forms 
    # (Currently constructs a rudimentary, buggy FST; your task is to implement a better one.
    f = buildFST()
    # Print out the FST translations of the input file
    f.parseInputFile(file)


    # f.addSetTransition("s",I,"x")

    # # c/o from x
    # f.addSetTransition("x",E,"q3")
    # f.addSetTransition("x",AZ-I-E,"s")
    # f.addSetTransition("x",I,"x")

    # # c/o from q3
    # f.addSetTransition("q3",I,"x")
    # f.addSetTransition("q3",AZ-I,"s")

    # f.addSetTransition("s",U,"z")
    # f.addSetTransition("s",CONS,"z")

    # # c/o of z
    # f.addSetTransition("z",E,"q1")
    # f.addSetTransition("z",VOWS,"y")
    # f.addSetTransition("z",CONS,"z")

    # f.addSetTransition("y",NPTR,"q2")


    # f.addTransition("q1","e","","ing")

    # f.addTransition("q2", "n", "nn", "ing")
    # f.addTransition("q2", "p", "pp", "ing")
    # f.addTransition("q2", "t", "tt", "ing")
    # f.addTransition("q2", "r", "rr", "ing")