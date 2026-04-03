/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import * as GenAiLib from '@google/genai';
import * as ClientLib from '@modelcontextprotocol/sdk/client/index.js';
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js';
import * as SdkClientStdioLib from '@modelcontextprotocol/sdk/client/stdio.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { AuthProviderType, type Config } from '../config/config.js';
import { GoogleCredentialProvider } from '../mcp/google-auth-provider.js';
import type { PromptRegistry } from '../prompts/prompt-registry.js';
import type { WorkspaceContext } from '../utils/workspaceContext.js';
import {
  connectAndDiscover,
  createTransport,
  hasNetworkTransport,
  isEnabled,
  McpClient,
  populateMcpServerCommand,
} from './mcp-client.js';
import type { ToolRegistry } from './tool-registry.js';

vi.mock('@modelcontextprotocol/sdk/client/stdio.js');
vi.mock('@modelcontextprotocol/sdk/client/index.js');
vi.mock('@google/genai');
vi.mock('../mcp/oauth-provider.js');
vi.mock('../mcp/oauth-token-storage.js');

describe('mcp-client', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('McpClient', () => {
    it('should discover tools', async () => {
      const mockedClient = {
        connect: vi.fn(),
        discover: vi.fn(),
        disconnect: vi.fn(),
        getStatus: vi.fn(),
        registerCapabilities: vi.fn(),
        setRequestHandler: vi.fn(),
        setNotificationHandler: vi.fn(),
      };
      vi.mocked(ClientLib.Client).mockReturnValue(
        mockedClient as unknown as ClientLib.Client,
      );
      vi.spyOn(SdkClientStdioLib, 'StdioClientTransport').mockReturnValue(
        {} as SdkClientStdioLib.StdioClientTransport,
      );
      const mockedMcpToTool = vi.mocked(GenAiLib.mcpToTool).mockReturnValue({
        tool: () => ({
          functionDeclarations: [
            {
              name: 'testFunction',
            },
          ],
        }),
      } as unknown as GenAiLib.CallableTool);
      const mockedToolRegistry = {
        registerTool: vi.fn(),
      } as unknown as ToolRegistry;
      const client = new McpClient(
        'test-server',
        {
          command: 'test-command',
        },
        mockedToolRegistry,
        {} as PromptRegistry,
        {} as WorkspaceContext,
        false,
      );
      await client.connect();
      await client.discover({} as Config);
      expect(mockedMcpToTool).toHaveBeenCalledOnce();
    });

    it('should not skip tools even if a parameter is missing a type', async () => {
      const mockedClient = {
        connect: vi.fn(),
        discover: vi.fn(),
        disconnect: vi.fn(),
        getStatus: vi.fn(),
        registerCapabilities: vi.fn(),
        setRequestHandler: vi.fn(),
        setNotificationHandler: vi.fn(),
        tool: vi.fn(),
      };
      vi.mocked(ClientLib.Client).mockReturnValue(
        mockedClient as unknown as ClientLib.Client,
      );
      vi.spyOn(SdkClientStdioLib, 'StdioClientTransport').mockReturnValue(
        {} as SdkClientStdioLib.StdioClientTransport,
      );
      vi.mocked(GenAiLib.mcpToTool).mockReturnValue({
        tool: () =>
          Promise.resolve({
            functionDeclarations: [
              {
                name: 'validTool',
                parametersJsonSchema: {
                  type: 'object',
                  properties: {
                    param1: { type: 'string' },
                  },
                },
              },
              {
                name: 'invalidTool',
                parametersJsonSchema: {
                  type: 'object',
                  properties: {
                    param1: { description: 'a param with no type' },
                  },
                },
              },
            ],
          }),
      } as unknown as GenAiLib.CallableTool);
      const mockedToolRegistry = {
        registerTool: vi.fn(),
      } as unknown as ToolRegistry;
      const client = new McpClient(
        'test-server',
        {
          command: 'test-command',
        },
        mockedToolRegistry,
        {} as PromptRegistry,
        {} as WorkspaceContext,
        false,
      );
      await client.connect();
      await client.discover({} as Config);
      expect(mockedToolRegistry.registerTool).toHaveBeenCalledTimes(2);
    });

    it('should handle errors when discovering prompts', async () => {
      const mockedClient = {
        connect: vi.fn(),
        discover: vi.fn(),
        disconnect: vi.fn(),
        getStatus: vi.fn(),
        registerCapabilities: vi.fn(),
        setRequestHandler: vi.fn(),
        setNotificationHandler: vi.fn(),
        getServerCapabilities: vi.fn().mockReturnValue({ prompts: {} }),
        request: vi.fn().mockRejectedValue(new Error('Test error')),
      };
      vi.mocked(ClientLib.Client).mockReturnValue(
        mockedClient as unknown as ClientLib.Client,
      );
      vi.spyOn(SdkClientStdioLib, 'StdioClientTransport').mockReturnValue(
        {} as SdkClientStdioLib.StdioClientTransport,
      );
      vi.mocked(GenAiLib.mcpToTool).mockReturnValue({
        tool: () => Promise.resolve({ functionDeclarations: [] }),
      } as unknown as GenAiLib.CallableTool);
      const client = new McpClient(
        'test-server',
        {
          command: 'test-command',
        },
        {} as ToolRegistry,
        {} as PromptRegistry,
        {} as WorkspaceContext,
        false,
      );
      await client.connect();
      await expect(client.discover({} as Config)).rejects.toThrow(
        'No prompts or tools found on the server.',
      );
    });
  });
  describe('appendMcpServerCommand', () => {
    it('should do nothing if no MCP servers or command are configured', () => {
      const out = populateMcpServerCommand({}, undefined);
      expect(out).toEqual({});
    });

    it('should discover tools via mcpServerCommand', () => {
      const commandString = 'command --arg1 value1';
      const out = populateMcpServerCommand({}, commandString);
      expect(out).toEqual({
        mcp: {
          command: 'command',
          args: ['--arg1', 'value1'],
        },
      });
    });

    it('should handle error if mcpServerCommand parsing fails', () => {
      expect(() => populateMcpServerCommand({}, 'derp && herp')).toThrowError();
    });
  });

  describe('createTransport', () => {
    describe('should connect via httpUrl', () => {
      it('without headers', async () => {
        const transport = await createTransport(
          'test-server',
          {
            httpUrl: 'http://test-server',
          },
          false,
        );

        expect(transport).toBeInstanceOf(StreamableHTTPClientTransport);
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        expect((transport as any)._url).toEqual(new URL('http://test-server'));
      });

      it('with headers', async () => {
        const transport = await createTransport(
          'test-server',
          {
            httpUrl: 'http://test-server',
            headers: { Authorization: 'derp' },
          },
          false,
        );

        expect(transport).toBeInstanceOf(StreamableHTTPClientTransport);
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        expect((transport as any)._url).toEqual(new URL('http://test-server'));
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        expect((transport as any)._requestInit?.headers).toEqual({
          Authorization: 'derp',
        });
      });
    });

    describe('should connect via url', () => {
      it('without headers', async () => {
        const transport = await createTransport(
          'test-server',
          {
            url: 'http://test-server',
          },
          false,
        );
        expect(transport).toBeInstanceOf(SSEClientTransport);
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        expect((transport as any)._url).toEqual(new URL('http://test-server'));
      });

      it('with headers', async () => {
        const transport = await createTransport(
          'test-server',
          {
            url: 'http://test-server',
            headers: { Authorization: 'derp' },
          },
          false,
        );

        expect(transport).toBeInstanceOf(SSEClientTransport);
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        expect((transport as any)._url).toEqual(new URL('http://test-server'));
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        expect((transport as any)._requestInit?.headers).toEqual({
          Authorization: 'derp',
        });
      });
    });

    it('should connect via command', async () => {
      const mockedTransport = vi
        .spyOn(SdkClientStdioLib, 'StdioClientTransport')
        .mockReturnValue({} as SdkClientStdioLib.StdioClientTransport);

      await createTransport(
        'test-server',
        {
          command: 'test-command',
          args: ['--foo', 'bar'],
          env: { FOO: 'bar' },
          cwd: 'test/cwd',
        },
        false,
      );

      expect(mockedTransport).toHaveBeenCalledWith({
        command: 'test-command',
        args: ['--foo', 'bar'],
        cwd: 'test/cwd',
        env: { ...process.env, FOO: 'bar' },
        stderr: 'pipe',
      });
    });

    describe('useGoogleCredentialProvider', () => {
      it('should use GoogleCredentialProvider when specified', async () => {
        const transport = await createTransport(
          'test-server',
          {
            httpUrl: 'http://test.googleapis.com',
            authProviderType: AuthProviderType.GOOGLE_CREDENTIALS,
            oauth: {
              scopes: ['scope1'],
            },
          },
          false,
        );

        expect(transport).toBeInstanceOf(StreamableHTTPClientTransport);
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const authProvider = (transport as any)._authProvider;
        expect(authProvider).toBeInstanceOf(GoogleCredentialProvider);
      });

      it('should use GoogleCredentialProvider with SSE transport', async () => {
        const transport = await createTransport(
          'test-server',
          {
            url: 'http://test.googleapis.com',
            authProviderType: AuthProviderType.GOOGLE_CREDENTIALS,
            oauth: {
              scopes: ['scope1'],
            },
          },
          false,
        );

        expect(transport).toBeInstanceOf(SSEClientTransport);
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const authProvider = (transport as any)._authProvider;
        expect(authProvider).toBeInstanceOf(GoogleCredentialProvider);
      });

      it('should throw an error if no URL is provided with GoogleCredentialProvider', async () => {
        await expect(
          createTransport(
            'test-server',
            {
              authProviderType: AuthProviderType.GOOGLE_CREDENTIALS,
              oauth: {
                scopes: ['scope1'],
              },
            },
            false,
          ),
        ).rejects.toThrow(
          'URL must be provided in the config for Google Credentials provider',
        );
      });
    });
  });
  describe('isEnabled', () => {
    const funcDecl = { name: 'myTool' };
    const serverName = 'myServer';

    it('should return true if no include or exclude lists are provided', () => {
      const mcpServerConfig = {};
      expect(isEnabled(funcDecl, serverName, mcpServerConfig)).toBe(true);
    });

    it('should return false if the tool is in the exclude list', () => {
      const mcpServerConfig = { excludeTools: ['myTool'] };
      expect(isEnabled(funcDecl, serverName, mcpServerConfig)).toBe(false);
    });

    it('should return true if the tool is in the include list', () => {
      const mcpServerConfig = { includeTools: ['myTool'] };
      expect(isEnabled(funcDecl, serverName, mcpServerConfig)).toBe(true);
    });

    it('should return true if the tool is in the include list with parentheses', () => {
      const mcpServerConfig = { includeTools: ['myTool()'] };
      expect(isEnabled(funcDecl, serverName, mcpServerConfig)).toBe(true);
    });

    it('should return false if the include list exists but does not contain the tool', () => {
      const mcpServerConfig = { includeTools: ['anotherTool'] };
      expect(isEnabled(funcDecl, serverName, mcpServerConfig)).toBe(false);
    });

    it('should return false if the tool is in both the include and exclude lists', () => {
      const mcpServerConfig = {
        includeTools: ['myTool'],
        excludeTools: ['myTool'],
      };
      expect(isEnabled(funcDecl, serverName, mcpServerConfig)).toBe(false);
    });

    it('should return false if the function declaration has no name', () => {
      const namelessFuncDecl = {};
      const mcpServerConfig = {};
      expect(isEnabled(namelessFuncDecl, serverName, mcpServerConfig)).toBe(
        false,
      );
    });
  });

  describe('hasNetworkTransport', () => {
    it('should return true if only url is provided', () => {
      const config = { url: 'http://example.com' };
      expect(hasNetworkTransport(config)).toBe(true);
    });

    it('should return true if only httpUrl is provided', () => {
      const config = { httpUrl: 'http://example.com' };
      expect(hasNetworkTransport(config)).toBe(true);
    });

    it('should return true if both url and httpUrl are provided', () => {
      const config = {
        url: 'http://example.com/sse',
        httpUrl: 'http://example.com/http',
      };
      expect(hasNetworkTransport(config)).toBe(true);
    });

    it('should return false if neither url nor httpUrl is provided', () => {
      const config = { command: 'do-something' };
      expect(hasNetworkTransport(config)).toBe(false);
    });

    it('should return false for an empty config object', () => {
      const config = {};
      expect(hasNetworkTransport(config)).toBe(false);
    });
  });

  describe('handleToolsListChanged', () => {
    it('should register setNotificationHandler during connect', async () => {
      const setNotificationHandler = vi.fn();
      const mockedClient = {
        connect: vi.fn(),
        disconnect: vi.fn(),
        getStatus: vi.fn(),
        registerCapabilities: vi.fn(),
        setRequestHandler: vi.fn(),
        setNotificationHandler,
      };
      vi.mocked(ClientLib.Client).mockReturnValue(
        mockedClient as unknown as ClientLib.Client,
      );
      vi.spyOn(SdkClientStdioLib, 'StdioClientTransport').mockReturnValue(
        {} as SdkClientStdioLib.StdioClientTransport,
      );

      const client = new McpClient(
        'test-server',
        { command: 'test' },
        { registerTool: vi.fn() } as unknown as ToolRegistry,
        {} as PromptRegistry,
        {} as WorkspaceContext,
        false,
      );

      await client.connect();

      expect(setNotificationHandler).toHaveBeenCalledTimes(1);
    });

    it('should register notification handler BEFORE client.connect() to catch early notifications', async () => {
      // This tests the race condition fix: if the MCP server sends
      // notifications/tools/list_changed during the connect handshake,
      // the handler must already be registered to receive it.
      const callOrder: string[] = [];
      const mockedClient = {
        connect: vi.fn().mockImplementation(async () => {
          callOrder.push('connect');
        }),
        disconnect: vi.fn(),
        getStatus: vi.fn().mockReturnValue('connected'),
        registerCapabilities: vi.fn(),
        setRequestHandler: vi.fn(),
        setNotificationHandler: vi.fn(() => {
          callOrder.push('setNotificationHandler');
        }),
      };
      vi.mocked(ClientLib.Client).mockReturnValue(
        mockedClient as unknown as ClientLib.Client,
      );
      vi.spyOn(SdkClientStdioLib, 'StdioClientTransport').mockReturnValue(
        {} as SdkClientStdioLib.StdioClientTransport,
      );

      const client = new McpClient(
        'test-server',
        { command: 'test' },
        { registerTool: vi.fn() } as unknown as ToolRegistry,
        {} as PromptRegistry,
        {} as WorkspaceContext,
        false,
      );

      await client.connect();

      // Notification handler MUST be registered before connect() is called
      expect(callOrder).toEqual(['setNotificationHandler', 'connect']);
    });

    it('should remove and re-register tools when notification handler fires', async () => {
      const removeMcpToolsByServer = vi.fn();
      const registerTool = vi.fn();
      const onToolsChanged = vi.fn();
      let capturedHandler: (() => Promise<void>) | undefined;
      const mockedClient = {
        connect: vi.fn(),
        disconnect: vi.fn(),
        getStatus: vi.fn().mockReturnValue('connected'),
        registerCapabilities: vi.fn(),
        setRequestHandler: vi.fn(),
        setNotificationHandler: vi.fn((_schema, handler) => {
          // Capture the handler so we can invoke it manually
          capturedHandler = handler;
        }),
        listTools: vi.fn().mockResolvedValue({
          tools: [
            {
              name: 'dynamicTool',
              description: 'A dynamically registered tool',
              inputSchema: { type: 'object', properties: {} },
            },
          ],
        }),
      };
      vi.mocked(ClientLib.Client).mockReturnValue(
        mockedClient as unknown as ClientLib.Client,
      );
      vi.spyOn(SdkClientStdioLib, 'StdioClientTransport').mockReturnValue(
        {} as SdkClientStdioLib.StdioClientTransport,
      );
      vi.mocked(GenAiLib.mcpToTool).mockReturnValue({
        tool: () =>
          Promise.resolve({
            functionDeclarations: [
              {
                name: 'dynamicTool',
                description: 'A dynamically registered tool',
                parametersJsonSchema: { type: 'object', properties: {} },
              },
            ],
          }),
      } as unknown as GenAiLib.CallableTool);

      const mockCliConfig = {
        isMcpServerDisabled: () => false,
        getExcludedMcpServers: () => [],
      } as unknown as Config;

      const client = new McpClient(
        'test-server',
        { command: 'test' },
        {
          registerTool,
          removeMcpToolsByServer,
        } as unknown as ToolRegistry,
        {} as PromptRegistry,
        {} as WorkspaceContext,
        false,
        undefined,
        mockCliConfig,
        onToolsChanged,
      );

      await client.connect();

      // Simulate the server sending a tools/list_changed notification
      expect(capturedHandler).toBeDefined();
      await capturedHandler!();

      // Verify tools were removed and re-registered
      expect(removeMcpToolsByServer).toHaveBeenCalledWith('test-server');
      expect(registerTool).toHaveBeenCalled();
      expect(onToolsChanged).toHaveBeenCalled();
    });

    it('should skip concurrent tool refreshes when handler fires twice rapidly', async () => {
      const removeMcpToolsByServer = vi.fn();
      const registerTool = vi.fn();
      let capturedHandler: (() => Promise<void>) | undefined;
      const mockedClient = {
        connect: vi.fn(),
        disconnect: vi.fn(),
        getStatus: vi.fn().mockReturnValue('connected'),
        registerCapabilities: vi.fn(),
        setRequestHandler: vi.fn(),
        setNotificationHandler: vi.fn((_schema, handler) => {
          capturedHandler = handler;
        }),
        listTools: vi.fn().mockResolvedValue({
          tools: [
            {
              name: 'dynamicTool',
              description: 'A dynamically registered tool',
              inputSchema: { type: 'object', properties: {} },
            },
          ],
        }),
      };
      vi.mocked(ClientLib.Client).mockReturnValue(
        mockedClient as unknown as ClientLib.Client,
      );
      vi.spyOn(SdkClientStdioLib, 'StdioClientTransport').mockReturnValue(
        {} as SdkClientStdioLib.StdioClientTransport,
      );

      const mockCliConfig = {
        isMcpServerDisabled: () => false,
        getExcludedMcpServers: () => [],
      } as unknown as Config;

      const client = new McpClient(
        'test-server',
        { command: 'test' },
        {
          registerTool,
          removeMcpToolsByServer,
        } as unknown as ToolRegistry,
        {} as PromptRegistry,
        {} as WorkspaceContext,
        false,
        undefined,
        mockCliConfig,
        vi.fn(),
      );

      await client.connect();

      // Fire the handler twice concurrently
      expect(capturedHandler).toBeDefined();
      await Promise.all([capturedHandler!(), capturedHandler!()]);

      // removeMcpToolsByServer should only be called once (not twice)
      // because the second call should be skipped due to the guard
      expect(removeMcpToolsByServer).toHaveBeenCalledTimes(1);
    });

    it('should handle notification gracefully when disconnected', async () => {
      const removeMcpToolsByServer = vi.fn();
      let capturedHandler: (() => Promise<void>) | undefined;
      const mockedClient = {
        connect: vi.fn(),
        disconnect: vi.fn(),
        close: vi.fn(),
        getStatus: vi.fn().mockReturnValue('disconnected'),
        registerCapabilities: vi.fn(),
        setRequestHandler: vi.fn(),
        setNotificationHandler: vi.fn((_schema, handler) => {
          capturedHandler = handler;
        }),
      };
      vi.mocked(ClientLib.Client).mockReturnValue(
        mockedClient as unknown as ClientLib.Client,
      );
      vi.spyOn(SdkClientStdioLib, 'StdioClientTransport').mockReturnValue({
        close: vi.fn(),
      } as unknown as SdkClientStdioLib.StdioClientTransport);

      const client = new McpClient(
        'test-server',
        { command: 'test' },
        {
          registerTool: vi.fn(),
          removeMcpToolsByServer,
        } as unknown as ToolRegistry,
        {} as PromptRegistry,
        {} as WorkspaceContext,
        false,
        undefined,
        {} as Config,
      );

      await client.connect();
      // Disconnect to set internal status back to DISCONNECTED
      await client.disconnect();

      // Simulate notification while disconnected
      expect(capturedHandler).toBeDefined();
      await capturedHandler!();

      // Should not attempt to remove tools when disconnected
      expect(removeMcpToolsByServer).not.toHaveBeenCalled();
    });

    it('should log error and continue when discoverTools fails during refresh', async () => {
      const removeMcpToolsByServer = vi.fn();
      const registerTool = vi.fn();
      const onToolsChanged = vi.fn();
      let capturedHandler: (() => Promise<void>) | undefined;
      const mockedClient = {
        connect: vi.fn(),
        disconnect: vi.fn(),
        getStatus: vi.fn().mockReturnValue('connected'),
        registerCapabilities: vi.fn(),
        setRequestHandler: vi.fn(),
        setNotificationHandler: vi.fn((_schema, handler) => {
          capturedHandler = handler;
        }),
        listTools: vi.fn().mockResolvedValue({
          tools: [
            {
              name: 'dynamicTool',
              description: 'A dynamically registered tool',
              inputSchema: { type: 'object', properties: {} },
            },
          ],
        }),
      };
      vi.mocked(ClientLib.Client).mockReturnValue(
        mockedClient as unknown as ClientLib.Client,
      );
      vi.spyOn(SdkClientStdioLib, 'StdioClientTransport').mockReturnValue(
        {} as SdkClientStdioLib.StdioClientTransport,
      );
      // mcpToTool throws on the refresh call (no initial discover in this test)
      vi.mocked(GenAiLib.mcpToTool).mockImplementation(() => {
        throw new Error('mcpToTool failed during refresh');
      });

      const mockCliConfig = {
        isMcpServerDisabled: () => false,
        getExcludedMcpServers: () => [],
      } as unknown as Config;

      const client = new McpClient(
        'test-server',
        { command: 'test' },
        {
          registerTool,
          removeMcpToolsByServer,
        } as unknown as ToolRegistry,
        {} as PromptRegistry,
        {} as WorkspaceContext,
        false,
        undefined,
        mockCliConfig,
        onToolsChanged,
      );

      await client.connect();

      // Simulate notification — mcpToTool will throw during refresh
      expect(capturedHandler).toBeDefined();
      await expect(capturedHandler!()).resolves.not.toThrow();

      // discoverTools catches the mcpToTool error and returns [],
      // so tools are removed but no new tools are registered.
      // onToolsChanged is still called because the error is caught inside discoverTools.
      expect(removeMcpToolsByServer).toHaveBeenCalledWith('test-server');
      expect(registerTool).not.toHaveBeenCalled();
      expect(onToolsChanged).toHaveBeenCalled();
    });
  });

  describe('connectAndDiscover notification handler', () => {
    it('should register notification handler BEFORE discoverPrompts/discoverTools to catch early notifications', async () => {
      const callOrder: string[] = [];
      let capturedHandler: (() => Promise<void>) | undefined;

      const mockedClient = {
        connect: vi.fn(),
        close: vi.fn(),
        onerror: null as ((error: Error) => void) | null,
        registerCapabilities: vi.fn(),
        setRequestHandler: vi.fn(),
        setNotificationHandler: vi.fn((_schema, handler) => {
          callOrder.push('setNotificationHandler');
          capturedHandler = handler;
        }),
        listTools: vi.fn().mockResolvedValue({
          tools: [
            {
              name: 'tool1',
              description: 'A tool',
              inputSchema: { type: 'object', properties: {} },
            },
          ],
        }),
        getServerCapabilities: vi.fn().mockReturnValue({ prompts: null }),
      };
      vi.mocked(ClientLib.Client).mockReturnValue(
        mockedClient as unknown as ClientLib.Client,
      );
      vi.spyOn(SdkClientStdioLib, 'StdioClientTransport').mockReturnValue(
        {} as SdkClientStdioLib.StdioClientTransport,
      );
      vi.mocked(GenAiLib.mcpToTool).mockReturnValue({
        tool: () =>
          Promise.resolve({
            functionDeclarations: [
              {
                name: 'tool1',
                description: 'A tool',
                parametersJsonSchema: { type: 'object', properties: {} },
              },
            ],
          }),
      } as unknown as GenAiLib.CallableTool);

      const mockCliConfig = {
        isMcpServerDisabled: () => false,
        getExcludedMcpServers: () => [],
        getMcpServers: () => ({}),
        getMcpServerCommand: () => undefined,
        isTrustedFolder: () => true,
      } as unknown as Config;

      const mockToolRegistry = {
        registerTool: vi.fn(),
        removeMcpToolsByServer: vi.fn(),
      } as unknown as ToolRegistry;

      const mockPromptRegistry = {
        clear: vi.fn(),
      } as unknown as PromptRegistry;

      await connectAndDiscover(
        'test-server',
        { command: 'test' },
        mockToolRegistry,
        mockPromptRegistry,
        false,
        {
          getDirectories: () => [],
          onDirectoriesChanged: vi.fn().mockReturnValue(() => {}),
        } as unknown as WorkspaceContext,
        mockCliConfig,
      );

      // Notification handler MUST be registered before any discovery calls
      expect(callOrder).toContain('setNotificationHandler');
      // The handler should be available (not undefined) after connectAndDiscover
      expect(capturedHandler).toBeDefined();
    });

    it('should register notification handler and refresh tools on tools/list_changed', async () => {
      const removeMcpToolsByServer = vi.fn();
      const registerTool = vi.fn();
      let capturedHandler: (() => Promise<void>) | undefined;

      const mockedClient = {
        connect: vi.fn(),
        close: vi.fn(),
        onerror: null as ((error: Error) => void) | null,
        registerCapabilities: vi.fn(),
        setRequestHandler: vi.fn(),
        setNotificationHandler: vi.fn((_schema, handler) => {
          capturedHandler = handler;
        }),
        listTools: vi.fn().mockResolvedValue({
          tools: [
            {
              name: 'refreshedTool',
              description: 'A refreshed tool',
              inputSchema: { type: 'object', properties: {} },
            },
          ],
        }),
        getServerCapabilities: vi.fn().mockReturnValue(null),
      };
      vi.mocked(ClientLib.Client).mockReturnValue(
        mockedClient as unknown as ClientLib.Client,
      );
      vi.spyOn(SdkClientStdioLib, 'StdioClientTransport').mockReturnValue(
        {} as SdkClientStdioLib.StdioClientTransport,
      );
      vi.mocked(GenAiLib.mcpToTool).mockReturnValue({
        tool: () =>
          Promise.resolve({
            functionDeclarations: [
              {
                name: 'refreshedTool',
                description: 'A refreshed tool',
                parametersJsonSchema: { type: 'object', properties: {} },
              },
            ],
          }),
      } as unknown as GenAiLib.CallableTool);

      const mockCliConfig = {
        isMcpServerDisabled: () => false,
        getExcludedMcpServers: () => [],
        getMcpServers: () => ({}),
        getMcpServerCommand: () => undefined,
        isTrustedFolder: () => true,
      } as unknown as Config;

      const mockToolRegistry = {
        registerTool,
        removeMcpToolsByServer,
      } as unknown as ToolRegistry;

      const mockPromptRegistry = {
        clear: vi.fn(),
      } as unknown as PromptRegistry;

      await connectAndDiscover(
        'test-server',
        { command: 'test' },
        mockToolRegistry,
        mockPromptRegistry,
        false,
        {
          getDirectories: () => [],
          onDirectoriesChanged: vi.fn().mockReturnValue(() => {}),
        } as unknown as WorkspaceContext,
        mockCliConfig,
      );

      // Verify handler was registered
      expect(capturedHandler).toBeDefined();

      // Simulate the server sending a tools/list_changed notification
      await capturedHandler!();

      // Verify tools were removed and re-registered
      expect(removeMcpToolsByServer).toHaveBeenCalledWith('test-server');
      expect(registerTool).toHaveBeenCalled();
    });

    it('should log error and continue when discoverTools fails during refresh', async () => {
      const removeMcpToolsByServer = vi.fn();
      const registerTool = vi.fn();
      let capturedHandler: (() => Promise<void>) | undefined;
      let mcpToToolCallCount = 0;

      const mockedClient = {
        connect: vi.fn(),
        close: vi.fn(),
        onerror: null as ((error: Error) => void) | null,
        registerCapabilities: vi.fn(),
        setRequestHandler: vi.fn(),
        setNotificationHandler: vi.fn((_schema, handler) => {
          capturedHandler = handler;
        }),
        listTools: vi.fn().mockResolvedValue({
          tools: [
            {
              name: 'initialTool',
              description: 'An initial tool',
              inputSchema: { type: 'object', properties: {} },
            },
          ],
        }),
        getServerCapabilities: vi.fn().mockReturnValue(null),
      };
      vi.mocked(ClientLib.Client).mockReturnValue(
        mockedClient as unknown as ClientLib.Client,
      );
      vi.spyOn(SdkClientStdioLib, 'StdioClientTransport').mockReturnValue(
        {} as SdkClientStdioLib.StdioClientTransport,
      );
      // First call to mcpToTool succeeds, second throws during refresh
      vi.mocked(GenAiLib.mcpToTool).mockImplementation(() => {
        mcpToToolCallCount++;
        if (mcpToToolCallCount === 1) {
          return {
            tool: () =>
              Promise.resolve({
                functionDeclarations: [
                  {
                    name: 'initialTool',
                    description: 'An initial tool',
                    parametersJsonSchema: { type: 'object', properties: {} },
                  },
                ],
              }),
          } as unknown as GenAiLib.CallableTool;
        }
        throw new Error('mcpToTool failed during refresh');
      });

      const mockCliConfig = {
        isMcpServerDisabled: () => false,
        getExcludedMcpServers: () => [],
        getMcpServers: () => ({}),
        getMcpServerCommand: () => undefined,
        isTrustedFolder: () => true,
      } as unknown as Config;

      const mockToolRegistry = {
        registerTool,
        removeMcpToolsByServer,
      } as unknown as ToolRegistry;

      const mockPromptRegistry = {
        clear: vi.fn(),
      } as unknown as PromptRegistry;

      await connectAndDiscover(
        'test-server',
        { command: 'test' },
        mockToolRegistry,
        mockPromptRegistry,
        false,
        {
          getDirectories: () => [],
          onDirectoriesChanged: vi.fn().mockReturnValue(() => {}),
        } as unknown as WorkspaceContext,
        mockCliConfig,
      );

      // Simulate the server sending a tools/list_changed notification
      // where re-discovery fails
      expect(capturedHandler).toBeDefined();
      await expect(capturedHandler!()).resolves.not.toThrow();

      // Tools were removed but not re-registered due to error
      expect(removeMcpToolsByServer).toHaveBeenCalledWith('test-server');
      expect(registerTool).toHaveBeenCalledTimes(1); // Only the initial registration
    });
  });
});
