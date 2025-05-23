package com.assignment.mlp;

import java.io.IOException;

public class UtilityFunctions {

    /**
     * Clears the console screen based on the operating system.
     * Uses ANSI escape codes for Unix-like systems and 'cls' command for Windows.
     *
     * @param osString The value of System.getProperty("os.name")
     */
    public static void clearConsole(String osString) {
        try {
            if (osString.contains("Windows")) {
                new ProcessBuilder("cmd", "/c", "cls").inheritIO().start().waitFor();
            } else {
                // Assume Unix-like system (Linux, macOS) and use ANSI escape codes
                System.out.print("\033[H\033[2J");
                System.out.flush();
            }
        } catch (IOException | InterruptedException e) {
            // Do nothing, or log the error. Clearing console is not critical.
            // System.err.println("Error clearing console: " + e.getMessage());
        }
    }
}