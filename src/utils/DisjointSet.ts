/**
 * DisjointSet utility with path compression. Some applications involve
 * grouping n distinct objects into a collection of disjoint sets. Two
 * important operations are then finding which set a given object belongs to
 * and uniting the two sets. A disjoint set data structure maintains a
 * collection S={ S1 , S2 ,..., Sk } of disjoint dynamic sets. Each set is
 * identified by a representative, which usually is a member in the set.
 */
export class DisjointSet {
    /**
     * Holds the length of the internal set.
     */
    public readonly length: number;

    /**
     * Holds the set containing the representative values.
     */
    private readonly parent: Uint32Array;

    /**
     * Creates a new DisjointSet instance.
     * @param length The length of the disjoint set
     * @throws Error if length is not specified
     */
    constructor(length: number) {
        if (length === undefined) {
            throw new Error("DisjointSet length not specified.");
        }
        this.length = length;
        this.parent = new Uint32Array(length);
        for (let i = 0; i < length; i++) {
            this.parent[i] = i;
        }
    }

    /**
     * Finds a pointer to the representative of the set containing i.
     * @param i The element to find the representative for
     * @returns The representative set of i
     */
    public find(i: number): number {
        if (this.parent[i] === i) {
            return i;
        } else {
            return (this.parent[i] = this.find(this.parent[i]));
        }
    }

    /**
     * Unites two dynamic sets containing objects i and j, say Si and Sj, into
     * a new set that Si ∪ Sj, assuming that Si ∩ Sj = ∅;
     * @param i First element
     * @param j Second element
     */
    public union(i: number, j: number): void {
        const iRepresentative = this.find(i);
        const jRepresentative = this.find(j);
        this.parent[iRepresentative] = jRepresentative;
    }
}
