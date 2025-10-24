       >>SOURCE FORMAT FREE
       IDENTIFICATION DIVISION.
       PROGRAM-ID. ST-DATASET-WRITER.
       ENVIRONMENT DIVISION.
       CONFIGURATION SECTION.
       SOURCE-COMPUTER. IBM-Z15.
       OBJECT-COMPUTER. IBM-Z15.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  WS-ROUTE.
           05  WS-DATASET-NAME        PIC X(44) VALUE 'ST.DATA.NARRATION(+1)'.
           05  WS-MEMBER              PIC X(8)  VALUE 'NARRATE '.
           05  WS-DISPOSITION         PIC X(3)  VALUE 'NEW'.
           05  WS-VOLUME              PIC X(6)  VALUE 'VOL001'.
           05  WS-RECORD-FORMAT       PIC X(4)  VALUE 'FB'.
           05  WS-RECORD-LENGTH       PIC 9(5)  VALUE 512.
           05  WS-BLOCK-SIZE          PIC 9(5)  VALUE 6144.
           05  WS-DATA-CLASS          PIC X(8)  VALUE 'NARRATE'.
           05  WS-MANAGEMENT-CLASS    PIC X(8)  VALUE 'GDG'.
           05  WS-STORAGE-CLASS       PIC X(8)  VALUE 'FASTIO'.
           05  WS-BUFFER-COUNT        PIC 9(4)  VALUE 0008.
           05  WS-KEY-LENGTH          PIC 9(4)  VALUE 0016.
           05  WS-KEY-OFFSET          PIC 9(4)  VALUE 0000.
           05  WS-CCSID               PIC 9(5)  VALUE 01208.
           05  WS-COMPRESS-DATA       PIC X     VALUE 'Y'.
           05  WS-SPACE-PRIMARY       PIC 9(5)  VALUE 15.
           05  WS-SPACE-SECONDARY     PIC 9(5)  VALUE 5.
           05  WS-SPACE-UNIT          PIC X(3)  VALUE 'CYL'.
           05  WS-DIRECTORY-BLOCKS    PIC 9(5)  VALUE 30.
           05  WS-DATASET-TYPE        PIC X(8)  VALUE 'LIBRARY'.
           05  WS-LIKE-DATASET        PIC X(44) VALUE 'ST.DATA.TEMPLATE'.
           05  WS-UNIT                PIC X(8)  VALUE 'SYSDA'.
           05  WS-AVGREC              PIC X(1)  VALUE 'K'.
           05  WS-RETENTION           PIC 9(4)  VALUE 0045.
           05  WS-RELEASE-SPACE       PIC X     VALUE 'Y'.
           05  WS-EXPIRATION          PIC 9(7)  VALUE 2025123.
       01  WS-DSORG                  PIC X(2)  VALUE SPACES.
       01  WS-TARGET-DSN             PIC X(64) VALUE SPACES.
       01  WS-TARGET-POINTER         PIC S9(4) COMP VALUE 1.
       01  WS-ALLOC-CMD              PIC X(256) VALUE SPACES.
       01  WS-ALLOC-POINTER          PIC S9(4) COMP VALUE 1.
       01  WS-LENGTH-TEXT            PIC Z(5)   VALUE ZEROES.
       01  WS-BLOCK-TEXT             PIC Z(5)   VALUE ZEROES.
       01  WS-BUFFER-TEXT            PIC Z(5)   VALUE ZEROES.
       01  WS-KEYLEN-TEXT            PIC Z(5)   VALUE ZEROES.
       01  WS-KEYOFF-TEXT            PIC Z(5)   VALUE ZEROES.
       01  WS-CCSID-TEXT             PIC 9(5)   VALUE ZEROES.
       01  WS-PRIMARY-TEXT           PIC 9(5)   VALUE ZEROES.
       01  WS-SECONDARY-TEXT         PIC 9(5)   VALUE ZEROES.
       01  WS-DIRECTORY-TEXT         PIC 9(5)   VALUE ZEROES.
       01  WS-RETENTION-TEXT         PIC Z(4)   VALUE ZEROES.
       01  WS-EXPIRATION-TEXT        PIC 9(7)   VALUE ZEROES.
       01  WS-RETURN-CODE            PIC S9(9) COMP VALUE ZERO.
       01  WS-MESSAGE                PIC X(80) VALUE SPACES.

       PROCEDURE DIVISION.
           *> Determine DSORG based on whether a PDS member was supplied.
           IF FUNCTION LENGTH(FUNCTION TRIM(WS-MEMBER)) > 0
               MOVE 'PO' TO WS-DSORG
           ELSE
               MOVE 'PS' TO WS-DSORG
           END-IF

           *> Ensure the block size can hold an integral number of records.
           IF WS-BLOCK-SIZE REM WS-RECORD-LENGTH NOT = 0
               MOVE 'Invalid block size for the supplied record length.' TO WS-MESSAGE
               DISPLAY WS-MESSAGE
               STOP RUN
           END-IF

           *> Build the fully-qualified dataset name, appending the member when present.
           MOVE SPACES TO WS-TARGET-DSN
           MOVE 1 TO WS-TARGET-POINTER
           STRING
               FUNCTION TRIM(WS-DATASET-NAME)
               INTO WS-TARGET-DSN
               WITH POINTER WS-TARGET-POINTER
           END-STRING

           IF FUNCTION LENGTH(FUNCTION TRIM(WS-MEMBER)) > 0
               STRING
                   '('
                   FUNCTION TRIM(WS-MEMBER)
                   ')'
                   INTO WS-TARGET-DSN
                   WITH POINTER WS-TARGET-POINTER
               END-STRING
           END-IF

           *> Convert numeric DCB values to editable strings for BPXWDYN.
           MOVE WS-RECORD-LENGTH TO WS-LENGTH-TEXT
           MOVE WS-BLOCK-SIZE TO WS-BLOCK-TEXT
           MOVE WS-BUFFER-COUNT TO WS-BUFFER-TEXT
           MOVE WS-KEY-LENGTH TO WS-KEYLEN-TEXT
           MOVE WS-KEY-OFFSET TO WS-KEYOFF-TEXT
           MOVE WS-CCSID TO WS-CCSID-TEXT
           MOVE WS-SPACE-PRIMARY TO WS-PRIMARY-TEXT
           MOVE WS-SPACE-SECONDARY TO WS-SECONDARY-TEXT
           MOVE WS-DIRECTORY-BLOCKS TO WS-DIRECTORY-TEXT
           MOVE WS-RETENTION TO WS-RETENTION-TEXT
           MOVE WS-EXPIRATION TO WS-EXPIRATION-TEXT

           *> Assemble the BPXWDYN allocation command driven by the WASM planner metadata.
           MOVE SPACES TO WS-ALLOC-CMD
           MOVE 1 TO WS-ALLOC-POINTER
           STRING
               'ALLOC FI(NARRBUF) DA(''' DELIMITED BY SIZE
               FUNCTION TRIM(WS-TARGET-DSN) DELIMITED BY SIZE
               ''') ' DELIMITED BY SIZE
               INTO WS-ALLOC-CMD
               WITH POINTER WS-ALLOC-POINTER
           END-STRING

           IF FUNCTION LENGTH(FUNCTION TRIM(WS-DISPOSITION)) > 0
               STRING
                   'DISP('
                   FUNCTION TRIM(WS-DISPOSITION)
                   ') '
                   INTO WS-ALLOC-CMD
                   WITH POINTER WS-ALLOC-POINTER
               END-STRING
           END-IF

           STRING
               'DSORG(' DELIMITED BY SIZE
               FUNCTION TRIM(WS-DSORG) DELIMITED BY SIZE
               ') ' DELIMITED BY SIZE
               'RECFM(' DELIMITED BY SIZE
               FUNCTION TRIM(WS-RECORD-FORMAT) DELIMITED BY SIZE
               ') ' DELIMITED BY SIZE
               'LRECL(' DELIMITED BY SIZE
               FUNCTION TRIM(WS-LENGTH-TEXT) DELIMITED BY SIZE
               ') ' DELIMITED BY SIZE
               'BLKSIZE(' DELIMITED BY SIZE
               FUNCTION TRIM(WS-BLOCK-TEXT) DELIMITED BY SIZE
               ') ' DELIMITED BY SIZE
               INTO WS-ALLOC-CMD
               WITH POINTER WS-ALLOC-POINTER
           END-STRING

           IF WS-BUFFER-COUNT > 0
               STRING
                   'BUFNO(' DELIMITED BY SIZE
                   FUNCTION TRIM(WS-BUFFER-TEXT) DELIMITED BY SIZE
                   ') ' DELIMITED BY SIZE
                   INTO WS-ALLOC-CMD
                   WITH POINTER WS-ALLOC-POINTER
               END-STRING
           END-IF

           IF WS-KEY-LENGTH > 0
               STRING
                   'KEYLEN(' DELIMITED BY SIZE
                   FUNCTION TRIM(WS-KEYLEN-TEXT) DELIMITED BY SIZE
                   ') ' DELIMITED BY SIZE
                   'KEYOFF(' DELIMITED BY SIZE
                   FUNCTION TRIM(WS-KEYOFF-TEXT) DELIMITED BY SIZE
                   ') ' DELIMITED BY SIZE
                   INTO WS-ALLOC-CMD
                   WITH POINTER WS-ALLOC-POINTER
               END-STRING
           END-IF

           IF FUNCTION LENGTH(FUNCTION TRIM(WS-VOLUME)) > 0
               STRING
                   'VOL(' DELIMITED BY SIZE
                   FUNCTION TRIM(WS-VOLUME) DELIMITED BY SIZE
                   ') ' DELIMITED BY SIZE
                   INTO WS-ALLOC-CMD
                   WITH POINTER WS-ALLOC-POINTER
               END-STRING
           END-IF

           IF FUNCTION LENGTH(FUNCTION TRIM(WS-DATA-CLASS)) > 0
               STRING
                   'DATACLAS(' DELIMITED BY SIZE
                   FUNCTION TRIM(WS-DATA-CLASS) DELIMITED BY SIZE
                   ') ' DELIMITED BY SIZE
                   INTO WS-ALLOC-CMD
                   WITH POINTER WS-ALLOC-POINTER
               END-STRING
           END-IF

           IF FUNCTION LENGTH(FUNCTION TRIM(WS-MANAGEMENT-CLASS)) > 0
               STRING
                   'MGMTCLAS(' DELIMITED BY SIZE
                   FUNCTION TRIM(WS-MANAGEMENT-CLASS) DELIMITED BY SIZE
                   ') ' DELIMITED BY SIZE
                   INTO WS-ALLOC-CMD
                   WITH POINTER WS-ALLOC-POINTER
               END-STRING
           END-IF

           IF FUNCTION LENGTH(FUNCTION TRIM(WS-STORAGE-CLASS)) > 0
               STRING
                   'STORCLAS(' DELIMITED BY SIZE
                   FUNCTION TRIM(WS-STORAGE-CLASS) DELIMITED BY SIZE
                   ') ' DELIMITED BY SIZE
                   INTO WS-ALLOC-CMD
                   WITH POINTER WS-ALLOC-POINTER
               END-STRING
           END-IF

           IF WS-SPACE-PRIMARY > 0 OR WS-SPACE-SECONDARY > 0
               STRING
                   'SPACE((' DELIMITED BY SIZE
                   FUNCTION TRIM(WS-PRIMARY-TEXT) DELIMITED BY SIZE
                   ' ' DELIMITED BY SIZE
                   FUNCTION TRIM(WS-SECONDARY-TEXT) DELIMITED BY SIZE
                   ')' DELIMITED BY SIZE
                   INTO WS-ALLOC-CMD
                   WITH POINTER WS-ALLOC-POINTER
               END-STRING
               IF FUNCTION LENGTH(FUNCTION TRIM(WS-SPACE-UNIT)) > 0
                   STRING
                       ' ' DELIMITED BY SIZE
                       FUNCTION TRIM(WS-SPACE-UNIT) DELIMITED BY SIZE
                       ') ' DELIMITED BY SIZE
                       INTO WS-ALLOC-CMD
                       WITH POINTER WS-ALLOC-POINTER
                   END-STRING
               ELSE
                   STRING
                       ') ' DELIMITED BY SIZE
                       INTO WS-ALLOC-CMD
                       WITH POINTER WS-ALLOC-POINTER
                   END-STRING
               END-IF
           END-IF

           IF WS-DIRECTORY-BLOCKS > 0
               STRING
                   'DIR(' DELIMITED BY SIZE
                   FUNCTION TRIM(WS-DIRECTORY-TEXT) DELIMITED BY SIZE
                   ') ' DELIMITED BY SIZE
                   INTO WS-ALLOC-CMD
                   WITH POINTER WS-ALLOC-POINTER
               END-STRING
           END-IF

           IF FUNCTION LENGTH(FUNCTION TRIM(WS-DATASET-TYPE)) > 0
               STRING
                   'DSNTYPE(' DELIMITED BY SIZE
                   FUNCTION TRIM(WS-DATASET-TYPE) DELIMITED BY SIZE
                   ') ' DELIMITED BY SIZE
                   INTO WS-ALLOC-CMD
                   WITH POINTER WS-ALLOC-POINTER
               END-STRING
           END-IF

           IF WS-CCSID > 0
               STRING
                   'CCSID(' DELIMITED BY SIZE
                   FUNCTION TRIM(WS-CCSID-TEXT) DELIMITED BY SIZE
                   ') ' DELIMITED BY SIZE
                   INTO WS-ALLOC-CMD
                   WITH POINTER WS-ALLOC-POINTER
               END-STRING
           END-IF

            IF FUNCTION LENGTH(FUNCTION TRIM(WS-LIKE-DATASET)) > 0
                STRING
                    'LIKE(''' DELIMITED BY SIZE
                    FUNCTION TRIM(WS-LIKE-DATASET) DELIMITED BY SIZE
                    ''') ' DELIMITED BY SIZE
                    INTO WS-ALLOC-CMD
                    WITH POINTER WS-ALLOC-POINTER
                END-STRING
            END-IF

            IF FUNCTION LENGTH(FUNCTION TRIM(WS-UNIT)) > 0
                STRING
                    'UNIT(' DELIMITED BY SIZE
                    FUNCTION TRIM(WS-UNIT) DELIMITED BY SIZE
                    ') ' DELIMITED BY SIZE
                    INTO WS-ALLOC-CMD
                    WITH POINTER WS-ALLOC-POINTER
                END-STRING
            END-IF

            IF FUNCTION LENGTH(FUNCTION TRIM(WS-AVGREC)) > 0
                STRING
                    'AVGREC(' DELIMITED BY SIZE
                    FUNCTION TRIM(WS-AVGREC) DELIMITED BY SIZE
                    ') ' DELIMITED BY SIZE
                    INTO WS-ALLOC-CMD
                    WITH POINTER WS-ALLOC-POINTER
                END-STRING
            END-IF

            IF WS-RETENTION > 0
                STRING
                    'RETENTION(' DELIMITED BY SIZE
                    FUNCTION TRIM(WS-RETENTION-TEXT) DELIMITED BY SIZE
                    ') ' DELIMITED BY SIZE
                    INTO WS-ALLOC-CMD
                    WITH POINTER WS-ALLOC-POINTER
                END-STRING
            END-IF

            IF WS-COMPRESS-DATA = 'Y'
                STRING
                    'COMPRESS ' DELIMITED BY SIZE
                    INTO WS-ALLOC-CMD
                    WITH POINTER WS-ALLOC-POINTER
                END-STRING
            ELSE
                IF WS-COMPRESS-DATA = 'N'
                    STRING
                        'NOCOMPRESS ' DELIMITED BY SIZE
                        INTO WS-ALLOC-CMD
                        WITH POINTER WS-ALLOC-POINTER
                    END-STRING
                END-IF
            END-IF

            IF WS-RELEASE-SPACE = 'Y'
                STRING
                    'RLSE ' DELIMITED BY SIZE
                    INTO WS-ALLOC-CMD
                    WITH POINTER WS-ALLOC-POINTER
                END-STRING
            END-IF

            IF WS-EXPIRATION > 0
                STRING
                    'EXPDT(' DELIMITED BY SIZE
                    FUNCTION TRIM(WS-EXPIRATION-TEXT) DELIMITED BY SIZE
                    ') ' DELIMITED BY SIZE
                    INTO WS-ALLOC-CMD
                    WITH POINTER WS-ALLOC-POINTER
                END-STRING
            END-IF

           DISPLAY 'BPXWDYN command built from planner metadata:'
           DISPLAY '  ' FUNCTION TRIM(WS-ALLOC-CMD)

           *> In production the command would be passed to BPXWDYN to allocate
           *> the dataset before writing the narration payload. Here we just
           *> confirm the computed statement.
           STOP RUN.
