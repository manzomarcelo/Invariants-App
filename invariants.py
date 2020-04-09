#!/usr/bin/env python2

from Tkinter import *
from sympy import *
import plink


# constants

RIGHT_HAND  = "RH"
LEFT_HAND   = "LH"
HAND_VALUE  = { RIGHT_HAND: 1, LEFT_HAND: -1 }
INF = "INF"


# variables to represent the polynomials

A = Symbol('A')
a = Symbol('a')
z = Symbol('z')
t = Symbol('t')


# general functions

def sqr(x):
    '''
    @param  int x
    @return int
    '''
    return x*x

def dist(p, q):
    '''
    @param  MY_Point    p
    @param  MY_Point    q
    @return float
    '''
    return sqr(q.x - p.x) + sqr(q.y - p.y)


class MY_Vertex(object):

    def __init__(self, pid, x, y):
        '''
        @param  int pid
        @param  int x
        @param  int y
        '''
        self.pid = pid
        self.x, self.y = x, y

    def __repr__(self):
        return "M(%d, (%d, %d))" %(self.pid, self.x, self.y)

    def __hash__(self):
        return hash((self.pid, self.point()))

    def __eq__(self, other):
        if type(other) == MY_Vertex:
            return self.x == other.x and self.y == other.y
        return False

    def __ne__(self, other):
        if type(other) == MY_Vertex:
            return self.x != other.x or self.y != other.y
        return True

    def point(self):
        return (self.x, self.y)


class MY_Arrow(object):

    def __init__(self, start, end):
        '''
        @param  MY_Vertex   start
        @param  MY_Vertex   end
        '''
        self.start = start
        self.end = end
        self.dx = self.end.x - self.start.x
        self.dy = self.end.y - self.start.y
        try:
            self.slope = -1.0*self.dy/self.dx
        except ZeroDivisionError:
            self.slope = INF

    def __repr__(self):
        return "%s => %s [%f]" %(self.start, self.end, self.slope)

    def __hash__(self):
        return hash((self.start, self.end))

    def __eq__(self, other):
        if type(other) == MY_Arrow:
            return self.start == other.start and self.end == other.end
        return False

    def __ne__(self, other):
        if type(other) == MY_Arrow:
            return self.start != other.start or self.end != other.end
        return True

    def __invert__(self):
        return MY_Arrow(self.end, self.start)

    def __contains__(self, obj):
        if type(obj) == MY_Vertex:
            return self.start == obj or self.end == obj
        return False


class MY_Crossing(object):

    def __init__(self, over, under, point):
        '''
        @param  MY_Arrow    over
        @param  MY_Arrow    under
        @param  int         pid
        @param  MY_Vertex   point
        '''
        self.over = over
        self.under = under
        self.cross_point = point

    def __repr__(self):
        return "%s over %s at %s" %(self.over, self.under, self.cross_point)

    def __hash__(self):
        return hash((self.over, self.under))

    def __eq__(self, other):
        if type(other) == MY_Crossing:
            return self.over == other.over and self.under == other.under
        return False

    def __contains__(self, obj):
        if type(obj) == MY_Arrow:
            return self.over == obj or self.under == obj
        if type(obj) == MY_Vertex:
            return obj in self.over or obj in self.under
        return False

    def sign(self):
        '''
        @return string
        '''
        d = self.under.dx*self.over.dy - self.under.dy*self.over.dx
        if d > 0:
            return RIGHT_HAND
        return LEFT_HAND

    def reverse(self):
        self.over, self.under = self.under, self.over


class InvariantsApp(object):
    """
    Computes some invariants associated with the knot drawn.
    Current invariats:
    - Kauffman Polynomial X
    - Kauffman Polynomial F
    - Jones Polynomial
    - Thurston-Bennequin Number
    - Maslov Number (Rotation Number)
    """

    ValueA = "A"
    ValueB = "B"

    def __init__(self, root):
        root.title("Invariants App")
        root.geometry("220x170")

        self.LinkEditor = plink.LinkEditor(root)

        # load the default file of the trefoil
        self.LinkEditor.load("samples/trefoil_knot.lnk")

        btnBracket = Button(root, text="Kauffman Bracket", width=20)
        btnBracket.bind("<Button-1>", self.btnBracket_click)
        btnBracket.pack()

        btnKauffmanX = Button(root, text="Kauffman Polynomial X", width=20)
        btnKauffmanX.bind("<Button-1>", self.btnKauffmanX_click)
        btnKauffmanX.pack()

        btnKauffmanF = Button(root, text="Kauffman Polynomial F", width=20)
        btnKauffmanF.bind("<Button-1>", self.btnKauffmanF_click)
        btnKauffmanF.pack()

        btnJones = Button(root, text="Jones Polynomial", width=20)
        btnJones.bind("<Button-1>", self.btnJones_click)
        btnJones.pack()

        btnTB = Button(root, text="Thurston-Bennequin Number", width=20)
        btnTB.bind("<Button-1>", self.btnThurstonBennequin_click)
        btnTB.pack()

        btnRotationNumber = Button(root, text="Rotation Number", width=20)
        btnRotationNumber.bind("<Button-1>", self.btnRotationNumber_click)
        btnRotationNumber.pack()

        '''
        btnPrint = Button(root, text="Print")
        btnPrint.bind("<Button-1>", self.btnPrint_click)
        btnPrint.pack()
        '''

    ''' Events '''

    def btnBracket_click(self, event):
        if self.LinkEditor.state == "start_state":
            self.read_link()
            bracket = self.KauffmanBracketX()
            print("Kauffman Bracket: %s" %bracket)
            self.LinkEditor.write_text("Kauffman Bracket: %s" %bracket)

    def btnKauffmanX_click(self, event):
        if self.LinkEditor.state == "start_state":
            self.read_link()
            poly = self.KauffmanPolynomialX()
            print("Kauffman Polynomial X: %s" %poly)
            self.LinkEditor.write_text("Kauffman Polynomial X: %s" %poly)

    def btnKauffmanF_click(self, event):
        if self.LinkEditor.state == "start_state":
            self.read_link()
            poly = self.KauffmanPolynomialF()
            print("Kauffman Polynomial F: %s" %poly)
            self.LinkEditor.write_text("Kauffman Polynomial F: %s" %poly)

    def btnJones_click(self, event):
        if self.LinkEditor.state == "start_state":
            self.read_link()
            poly = self.JonesPolynomial()
            print("Jones Polynomial: %s" %poly)
            self.LinkEditor.write_text("Jones Polynomial: %s" %poly)

    def btnThurstonBennequin_click(self, event):
        if self.LinkEditor.state == "start_state":
            self.read_link()
            beta = self.ThurstonBennequinNumber()
            print("Thurston-Bennequin Number: %d" %beta)
            self.LinkEditor.write_text("Thurston-Bennequin Number: %s" %beta)

    def btnRotationNumber_click(self, event):
        if self.LinkEditor.state == "start_state":
            self.read_link()
            r = self.RotationNumber()
            print("Rotation Number: %d" %r)
            self.LinkEditor.write_text("Maslov Number: %s" %r)

    def btnPrint_click(self, event):
        '''
        Auxiliar (temporary) method for debugging
        '''
        if self.LinkEditor.state == "start_state":
            self.read_link()
            print(self.arrows)

    ''' Methods '''

    def print_graph(self):
        '''
        Auxiliar (temporary) method for debugging
        '''
        print("\t"*self.tab + "Graph:")
        for key, value in self.graph.items():
            print("\t"*self.tab + "%s : %s" %(key, value))

    def read_link(self):
        '''
        Initialize the current link.
        '''
        vertices = self.LinkEditor.Vertices
        arrows = self.LinkEditor.Arrows
        crossings = self.LinkEditor.Crossings

        self.qnt_vertices = len(vertices)
        self.qnt_crossings = len(crossings)
        self.qnt_points = self.qnt_vertices + self.qnt_crossings

        list_vertices = [((v.x, v.y), i) for i, v in enumerate(vertices)]
        list_crossings = [
            ((c.x, c.y), i) for i, c in enumerate(crossings, self.qnt_vertices)
        ]
        map_vertices = dict(list_vertices + list_crossings)

        # convert from Vertex object of plink module to MY_Vertex object
        my_vertices = [
            MY_Vertex(map_vertices[v.point()], v.x, v.y) for v in vertices
        ]
        self.vertices = my_vertices

        # convert from Arrow object of plink module to MY_Arrow object
        my_arrows = []
        for a in arrows:
            p = a.start.point()
            mp = MY_Vertex(map_vertices[p], *p)
            q = a.end.point()
            mq = MY_Vertex(map_vertices[q], *q)
            my_arrows.append(MY_Arrow(mp, mq))

        self.arrows = my_arrows

        # convert from Crossing object of plink module to MY_Crossing object
        my_crossings = []
        for c in crossings:
            p = c.over.start.point()
            mp = MY_Vertex(map_vertices[p], *p)
            q = c.over.end.point()
            mq = MY_Vertex(map_vertices[q], *q)
            over = MY_Arrow(mp, mq)

            p = c.under.start.point()
            mp = MY_Vertex(map_vertices[p], *p)
            q = c.under.end.point()
            mq = MY_Vertex(map_vertices[q], *q)
            under = MY_Arrow(mp, mq)

            p = (c.x, c.y)
            mp = MY_Vertex(map_vertices[p], *p)
            my_crossings.append(MY_Crossing(over, under, mp))

        self.crossings = my_crossings

        # build the graph representation of the link
        self.build_graph()

    def build_graph(self):
        '''
        Build the graph representation of the link.
        '''
        # add the vertices
        graph = dict([(v, []) for v in self.vertices])
        digraph = dict([(v, None) for v in self.vertices])

        # add the crossing points
        for crossing in self.crossings:
            cp = crossing.cross_point
            graph[cp] = []
            graph[MY_Vertex(-cp.pid, *cp.point())] = []

        # add the arrows
        for arrow in self.arrows:
            p = arrow.start
            q = arrow.end
            graph[p].append(q)
            graph[q].append(p)
            digraph[p] = q

        self.graph = graph
        self.digraph = digraph

    def count_components(self, graph):
        '''
        Counts how many components there are in the graph.
        @param  graph   map representation of the graph
        @return int
        '''
        def visit(v, graph):
            '''
            @param  int v
            @param  graph   map representation of the graph
            '''
            visited[v] = True
            for w in graph[v]:
                if not visited[w]:
                    visit(w, graph)

        qnt = 0
        visited = dict([(key, False) for key in graph.keys()])
        for key in graph.keys():
            if not visited[key]:
                qnt += 1
                visit(key, graph)

        return qnt

    def eval_writhe(self):
        '''
        Evaluate the writhe of the link.
        @return int
        '''
        writhe = 0
        for c in self.crossings:
            writhe += HAND_VALUE[c.sign()]
        return writhe

    def uncross(self, idx, crossings, value):
        '''
        Uncross the crossing point to the respective value.
        @param  idx         index of the crossing
        @param  crossings   list of crossings
        @param  value       value to be set in the uncrossing
        '''
        crossing = crossings[idx]
        cp = crossing.cross_point
        cpr = MY_Vertex(-cp.pid, *cp.point())
        os = crossing.over.start
        oe = crossing.over.end
        us = crossing.under.start
        ue = crossing.under.end
        sign = crossing.sign()

        dist_os_cp = dist(os, cp)
        dist_us_cp = dist(us, cp)

        # update the start/end point of all of
        # the crossings involved in the current crossing
        for c in crossings:
            if crossing == c:
                continue

            tmp = None
            if crossing.over == c.over:
                tmp = c.over
            elif crossing.over == c.under:
                tmp = c.under
            if tmp != None:
                if dist(os, c.cross_point) > dist_os_cp:
                    tmp.start = cpr
                else:
                    tmp.end = cp

            tmp = None
            if crossing.under == c.over:
                tmp = c.over
            elif crossing.under == c.under:
                tmp = c.under
            if tmp != None:
                if (sign == RIGHT_HAND and value == self.ValueA) or \
                   (sign == LEFT_HAND and value == self.ValueB):
                    if dist(us, c.cross_point) > dist_us_cp:
                        tmp.start = cp
                    else:
                        tmp.end = cpr
                else:
                    if dist(us, c.cross_point) > dist_us_cp:
                        tmp.start = cpr
                    else:
                        tmp.end = cp

        # change the origin point just for convinience
        # when setting the new graph according to the crossing value
        if sign == RIGHT_HAND and value == self.ValueB:
            us, ue = ue, us
        elif sign == LEFT_HAND and value == self.ValueA:
            us, ue = ue, us

        # update the graph
        self.graph[os].remove(oe)
        self.graph[oe].remove(os)
        self.graph[us].remove(ue)
        self.graph[ue].remove(us)

        self.graph[os].append(cp)
        self.graph[cp].append(os)
        self.graph[cp].append(ue)
        self.graph[ue].append(cp)
        self.graph[us].append(cpr)
        self.graph[cpr].append(us)
        self.graph[cpr].append(oe)
        self.graph[oe].append(cpr)

    def cross_back(self, idx, crossings, value):
        '''
        Cross back the crossing point to the previous state.
        @param  idx         index of the crossing
        @param  crossings   list of crossings
        @param  value       value that is set
        '''
        crossing = crossings[idx]
        cp = crossing.cross_point
        cpr = MY_Vertex(-cp.pid, *cp.point())
        os = crossing.over.start
        oe = crossing.over.end
        us = crossing.under.start
        ue = crossing.under.end
        sign = crossing.sign()

        # update back the start/end point of all of
        # the crossings involved in the current crossing
        for c in crossings:
            if crossing == c:
                continue

            tmp = None
            if cp == c.over.start:
                tmp = c.over
            elif cp == c.under.start:
                tmp = c.under
            if tmp != None:
                if tmp.end == oe:
                    tmp.start = os
                else:
                    tmp.start = us

            tmp = None
            if cp == c.over.end:
                tmp = c.over
            elif cp == c.under.end:
                tmp = c.under
            if tmp != None:
                if tmp.start == os:
                    tmp.end = oe
                else:
                    tmp.end = ue

        # change the origin point just for convinience
        # when setting the new graph according to the crossing value
        if sign == RIGHT_HAND and value == self.ValueB:
            us, ue = ue, us
        elif sign == LEFT_HAND and value == self.ValueA:
            us, ue = ue, us

        # update the graph
        self.graph[cp] = []
        self.graph[cpr] = []

        self.graph[os].remove(cp)
        self.graph[oe].remove(cpr)
        self.graph[us].remove(cpr)
        self.graph[ue].remove(cp)

        self.graph[os].append(oe)
        self.graph[oe].append(os)
        self.graph[us].append(ue)
        self.graph[ue].append(us)

    def KauffmanBracketX(self):
        '''
        Evaluate the Kauffman Bracket related to Kauffman Polynomial X.
        delta = -(A^2 + A^(-2))
        <L> = sum{A^(a(S) - b(S)) * delta^|S| : S are the states}
        '''
        def eval_bracket(idx, states):
            '''
            Evaluate the bracket polynomial.
            @param  idx     index of the crossing
            @param  states  list of states of the crossings
            '''
            if idx == self.qnt_crossings:
                qnt_components = self.count_components(self.graph)
                delta = -(A**2 + A**(-2))
                return A**sum(states) * delta**(qnt_components - 1)

            states[idx] = 1
            self.uncross(idx, self.crossings, self.ValueA)
            partial = eval_bracket(idx+1, states)
            self.cross_back(idx, self.crossings, self.ValueA)

            states[idx] = -1
            self.uncross(idx, self.crossings, self.ValueB)
            partial += eval_bracket(idx+1, states)
            self.cross_back(idx, self.crossings, self.ValueB)

            states[idx] = 0

            return partial

        bracket = eval_bracket(0, [0]*self.qnt_crossings)
        bracket = bracket.simplify().evalf()
        return bracket

    def KauffmanBracketF(self):
        '''
        Evaluate the Kauffman Bracket related to Kauffman Polynomial F.
        '''
        def create_crossing_order_list(crossings):
            '''
            Create the list of crossings in the order that it must be visited.
            @param  crossings   list of crossings
            @return list        list of crossings in order that must be visited
            '''
            crossing_order = []
            crossing_trash = []

            visited = dict(
                [(v, False) for v, l in self.graph.items() if l != []]
            )
            for v in visited.keys():
                if not visited[v]:
                    visited[v] = True
                    p = v
                    l = self.graph[p]
                    if not visited[l[0]]:
                        q = l[0]
                    elif not visited[l[1]]:
                        q = l[1]
                    else:
                        #TODO verify if it is necessary (i thing not)
                        continue
                    first = True
                    while p != v or first:
                        if first:
                            first = False
                        visited[q] = True
                        arrow = MY_Arrow(p, q)
                        arrowr = MY_Arrow(q, p)
                        # check for crossing over edge pq
                        tmp = []
                        for c in crossings:
                            if c not in crossing_trash:
                                if c.over == arrow or c.over == arrowr:
                                    crossing_trash.append(c)
                                elif c.under == arrow or c.under == arrowr:
                                    tmp.append(c)
                                    crossing_trash.append(c)
                        # sort in order of distance from endpoint p
                        tmp = sorted(tmp, key=lambda c: dist(p, c.cross_point))
                        crossing_order.extend(tmp)
                        # update nodes
                        p = q
                        l = self.graph[p]
                        if not visited[l[0]]:
                            q = l[0]
                        elif not visited[l[1]]:
                            q = l[1]
                        elif l[0] == v:
                            q = l[0]
                        elif l[1] == v:
                            q = l[1]
                        else:
                            break

            return crossing_order

        def eval_trivial_bracket(crossings):
            '''
            Evaluate the bracket of a link of trivial knots
            without crossing between them.
            @param  crossings   list of crossings
            @return bracket     bracket polynomial of a link of trivial knots
            '''
            delta = z**(-1) * (a + a**(-1)) - 1
            bracket = 1/delta

            visited = dict(
                [(v, False) for v, l in self.graph.items() if l != []]
            )
            for v in visited.keys():
                # for each component
                if not visited[v]:
                    visited[v] = True
                    p = v
                    l = self.graph[p]
                    if not visited[l[0]]:
                        q = l[0]
                    else:
                        q = l[1]
                    # list of vertices in the component
                    comp_vertices = [p]
                    # list of arrows in the component
                    comp_arrows = []
                    while True:
                        visited[q] = True
                        comp_vertices.append(q)
                        comp_arrows.append(MY_Arrow(p, q))
                        p = q
                        l = self.graph[p]
                        if not visited[l[0]]:
                            q = l[0]
                        elif not visited[l[1]]:
                            q = l[1]
                        else:
                            break
                    # appending the last vertex that it is the first one
                    comp_vertices.append(v)
                    # appending the last arrow that closes the component
                    comp_arrows.append(MY_Arrow(q, v))

                    writhe = 0
                    for c in crossings:
                        if c.over in comp_arrows and c.under in comp_arrows:
                            writhe += HAND_VALUE[c.sign()]
                        elif c.over in comp_arrows and ~c.under in comp_arrows:
                            c.under = ~c.under
                            writhe += HAND_VALUE[c.sign()]
                            c.under = ~c.under
                        elif ~c.over in comp_arrows and c.under in comp_arrows:
                            c.over = ~c.over
                            writhe += HAND_VALUE[c.sign()]
                            c.over = ~c.over
                        elif ~c.over in comp_arrows and ~c.under in comp_arrows:
                            writhe += HAND_VALUE[c.sign()]

                    partial = a**writhe
                    bracket *= delta * partial

            return bracket

        def eval_bracket(crossings, idx, crossing_order, qnt_crossings):
            '''
            Evaluate the bracket polynomial.
            @param  crossings       list of crossings
            @param  idx             index of the crossing
            @param  crossing_order  list of crossings to be modified
            @param  qnt_crossings   qnt of crossings to be modified
            '''
            if idx == qnt_crossings:
                return eval_trivial_bracket(crossings)

            # crossing index in the list of crossings
            crossing_idx = crossings.index(crossing_order[idx])

            # uncross to value A
            self.uncross(crossing_idx, crossings, self.ValueA)
            new_crossings = crossings[:]
            new_crossings.pop(crossing_idx)
            new_crossing_order = create_crossing_order_list(new_crossings)
            partial = eval_bracket(new_crossings, 0, new_crossing_order,
                                   len(new_crossing_order))
            self.cross_back(crossing_idx, crossings, self.ValueA)

            # uncross to value B
            self.uncross(crossing_idx, crossings, self.ValueB)
            new_crossings = crossings[:]
            new_crossings.pop(crossing_idx)
            new_crossing_order = create_crossing_order_list(new_crossings)
            partial += eval_bracket(new_crossings, 0, new_crossing_order,
                                    len(new_crossing_order))
            self.cross_back(crossing_idx, crossings, self.ValueB)

            # multiply to z variable
            partial *= z

            # change the crossing
            crossings[crossing_idx].reverse()
            partial -= eval_bracket(crossings, idx+1, crossing_order,
                                    qnt_crossings)
            crossings[crossing_idx].reverse()

            return partial

        crossing_order = create_crossing_order_list(self.crossings)
        bracket = eval_bracket(self.crossings, 0, crossing_order,
                               len(crossing_order))
        bracket = bracket.simplify().evalf()
        return bracket

    def KauffmanPolynomialX(self):
        '''
        Evaluate the Kauffman Polynomial X.
        '''
        writhe = self.eval_writhe()
        bracket = self.KauffmanBracketX()
        poly = (-A**3)**(-writhe) * bracket
        poly = poly.simplify().evalf()
        return poly

    def KauffmanPolynomialF(self):
        '''
        Evaluate the Kauffman Polynomial F.
        '''
        writhe = self.eval_writhe()
        bracket = self.KauffmanBracketF()
        poly = a**(-writhe) * bracket
        poly = poly.simplify().evalf()
        return poly

    def JonesPolynomial(self):
        '''
        Evaluate the Jones Polynomial.
        '''
        ### Compare results with Kauffman Polynomial F
        #tmp_aux = self.KauffmanPolynomialF();
        #tmp_poly = tmp_aux.subs(a, -t**(Rational(-3, 4)))
        #tmp_poly = tmp_poly.subs(z, t**(Rational(-1, 4)) + t**(Rational(1, 4)))
        #tmp_poly = tmp_poly.simplify().evalf()
        #print tmp_poly
        ###
        aux_poly = self.KauffmanPolynomialX()
        poly = aux_poly.subs(A, t**(Rational(-1, 4)))
        poly = poly.simplify().evalf();
        return poly

    def ThurstonBennequinNumber(self):
        '''
        Evaluate the Thurston-Bennequin Number of the link considering
        the basic modifications to aproximate a Legendrian link.
        '''
        cusps = 0

        for crossing in self.crossings:
            if crossing.under.slope < crossing.over.slope:
                cusps += 2

        visited = dict([(v, False) for v in self.vertices])
        for v in visited.keys():
            if not visited[v]:
                p = v
                q = self.digraph[p]
                visited[p] = visited[q] = True
                s = None
                while s != v:
                    s = q
                    t = self.digraph[s]
                    visited[s] = visited[t] = True
                    pq = MY_Arrow(p, q)
                    st = MY_Arrow(s, t)
                    #XXX it can fail if any dx or dy is equal to 0
                    if pq.dx * st.dx < 0 and pq.dy * st.dy > 0:
                        cusps += 1
                    p, q = s, t

        writhe = self.eval_writhe()
        beta = writhe - cusps/2
        return beta

    def RotationNumber(self):
        '''
        Evaluate the Rotation Number (Maslov Number) of the link considering
        the basic modifications to aproximate a Legendrian link.
        '''
        up_cusps = down_cusp = 0

        for crossing in self.crossings:
            if crossing.under.slope < crossing.over.slope:
                if crossing.under.dy > 0:
                    down_cusp += 2
                else:
                    up_cusps += 2

        visited = dict([(v, False) for v in self.vertices])
        for v in visited.keys():
            if not visited[v]:
                p = v
                q = self.digraph[p]
                visited[p] = visited[q] = True
                s = None
                while s != v:
                    s = q
                    t = self.digraph[s]
                    visited[s] = visited[t] = True
                    pq = MY_Arrow(p, q)
                    st = MY_Arrow(s, t)
                    #XXX it can fail if any dx or dy is equal to 0
                    if pq.dx * st.dx < 0:
                        if pq.dy > 0 and st.dy > 0:
                            down_cusp += 1
                        elif pq.dy < 0 and st.dy < 0:
                            up_cusps += 1
                    p, q = s, t

        r = (up_cusps - down_cusp)/2
        return r

def main():
    root = Tk()

    InvariantsApp(root)

    root.mainloop()

if __name__ == "__main__":
    main()
