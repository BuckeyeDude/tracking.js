/**
 * Math utility class for tracking operations.
 */
export class TrackingMath {
  /**
   * Euclidean distance between two points P(x0, y0) and P(x1, y1).
   * @param x0 Horizontal coordinate of P0.
   * @param y0 Vertical coordinate of P0.
   * @param x1 Horizontal coordinate of P1.
   * @param y1 Vertical coordinate of P1.
   * @return The euclidean distance.
   */
  static distance(x0: number, y0: number, x1: number, y1: number): number {
    const dx = x1 - x0;
    const dy = y1 - y0;

    return Math.sqrt(dx * dx + dy * dy);
  }

  /**
   * Calculates the Hamming weight of a string, which is the number of symbols that are
   * different from the zero-symbol of the alphabet used. It is thus
   * equivalent to the Hamming distance from the all-zero string of the same
   * length. For the most typical case, a string of bits, this is the number
   * of 1's in the string.
   *
   * Example:
   *
   * ```
   *  Binary string     Hamming weight
   *   11101                 4
   *   11101010              5
   * ```
   *
   * @param i Number that holds the binary string to extract the hamming weight.
   * @return The hamming weight.
   */
  static hammingWeight(i: number): number {
    i = i - ((i >> 1) & 0x55555555);
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333);

    return ((i + (i >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
  }

  /**
   * Generates a random number between [a, b] interval.
   * @param a Lower bound (inclusive)
   * @param b Upper bound (inclusive)
   * @return Random number between a and b
   */
  static uniformRandom(a: number, b: number): number {
    return a + Math.random() * (b - a);
  }

  /**
   * Tests if a rectangle intersects with another.
   *
   * ```
   *  x0y0 --------       x2y2 --------
   *      |       |           |       |
   *      -------- x1y1       -------- x3y3
   * ```
   *
   * @param x0 Horizontal coordinate of P0.
   * @param y0 Vertical coordinate of P0.
   * @param x1 Horizontal coordinate of P1.
   * @param y1 Vertical coordinate of P1.
   * @param x2 Horizontal coordinate of P2.
   * @param y2 Vertical coordinate of P2.
   * @param x3 Horizontal coordinate of P3.
   * @param y3 Vertical coordinate of P3.
   * @return True if rectangles intersect, false otherwise
   */
  static intersectRect(
    x0: number, 
    y0: number, 
    x1: number, 
    y1: number, 
    x2: number, 
    y2: number, 
    x3: number, 
    y3: number
  ): boolean {
    return !(x2 > x1 || x3 < x0 || y2 > y1 || y3 < y0);
  }
}